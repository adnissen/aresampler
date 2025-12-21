//! macOS audio capture implementation using ScreenCaptureKit

use crate::process::process_exists;
use crate::ring_buffer::AudioRingBuffer;
use crate::types::{
    CaptureCommand, CaptureConfig, CaptureEvent, CaptureStats, MonitorConfig, SourceStats,
};
use anyhow::{anyhow, Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use screencapturekit::cm::CMSampleBuffer;
use screencapturekit::shareable_content::SCShareableContent;
use screencapturekit::stream::configuration::SCStreamConfiguration;
use screencapturekit::stream::content_filter::SCContentFilter;
use screencapturekit::stream::output_trait::SCStreamOutputTrait;
use screencapturekit::stream::output_type::SCStreamOutputType;
use screencapturekit::stream::SCStream;
use std::collections::VecDeque;
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Audio source type for channel routing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AudioSourceType {
    App,
    Microphone,
}

/// Channel buffer that accumulates samples from multiple sources and interleaves them
/// to separate channels (app on channels 1-2, mic on channels 3-4)
struct AudioChannelBuffer {
    /// Accumulated app audio samples (interleaved stereo: L, R, L, R, ...)
    app_buffer: VecDeque<f32>,
    /// Accumulated microphone audio samples (interleaved stereo: L, R, L, R, ...)
    mic_buffer: VecDeque<f32>,
    /// Number of input channels per source (typically 2 for stereo)
    input_channels: usize,
    /// Number of output channels (2 if single source, 4 if both sources)
    output_channels: usize,
    /// Whether we're capturing app audio
    has_app_audio: bool,
    /// Whether we're capturing mic audio
    has_mic_audio: bool,
    /// Whether we've started outputting (after initial sync)
    started: bool,
    /// Minimum frames to buffer before starting (for initial sync)
    min_buffer_frames: usize,
}

impl AudioChannelBuffer {
    fn new(input_channels: usize, has_app_audio: bool, has_mic_audio: bool) -> Self {
        // Output channels: 2 per active source
        let output_channels = if has_app_audio && has_mic_audio {
            input_channels * 2 // 4 channels: app L/R + mic L/R
        } else {
            input_channels // 2 channels: just one source
        };

        // Buffer ~50ms of audio at 48kHz before starting output (for sync)
        // This gives both sources time to start delivering
        let min_buffer_frames = if has_app_audio && has_mic_audio {
            2400 // ~50ms at 48kHz
        } else {
            0 // No buffering needed for single source
        };

        Self {
            app_buffer: VecDeque::new(),
            mic_buffer: VecDeque::new(),
            input_channels,
            output_channels,
            has_app_audio,
            has_mic_audio,
            started: false,
            min_buffer_frames,
        }
    }

    /// Push samples from a specific source
    fn push(&mut self, source: AudioSourceType, samples: &[f32]) {
        match source {
            AudioSourceType::App => self.app_buffer.extend(samples),
            AudioSourceType::Microphone => self.mic_buffer.extend(samples),
        }
    }

    /// Drain and interleave available samples from all active sources
    /// Returns interleaved samples: [app_L, app_R, mic_L, mic_R] per frame when both active
    fn drain_interleaved(&mut self) -> Vec<f32> {
        // If only one source is active, just drain that source (2-channel output)
        if self.has_app_audio && !self.has_mic_audio {
            return self.app_buffer.drain(..).collect();
        }
        if self.has_mic_audio && !self.has_app_audio {
            return self.mic_buffer.drain(..).collect();
        }

        // Both sources active - interleave to 4 channels
        let app_frames = self.app_buffer.len() / self.input_channels;
        let mic_frames = self.mic_buffer.len() / self.input_channels;

        // Wait for both sources to have minimum buffer before starting
        if !self.started {
            if app_frames >= self.min_buffer_frames && mic_frames >= self.min_buffer_frames {
                self.started = true;
            } else {
                return Vec::new();
            }
        }

        // Once started, output the minimum of both sources to stay synchronized
        let available_frames = app_frames.min(mic_frames);

        if available_frames == 0 {
            return Vec::new();
        }

        // Output: 4 samples per frame (app_L, app_R, mic_L, mic_R)
        let mut interleaved = Vec::with_capacity(available_frames * self.output_channels);

        for _ in 0..available_frames {
            // Pop one frame from app (L, R)
            let app_l = self.app_buffer.pop_front().unwrap_or(0.0);
            let app_r = self.app_buffer.pop_front().unwrap_or(0.0);
            // Pop one frame from mic (L, R)
            let mic_l = self.mic_buffer.pop_front().unwrap_or(0.0);
            let mic_r = self.mic_buffer.pop_front().unwrap_or(0.0);
            // Interleave: app channels first, then mic channels
            interleaved.push(app_l);
            interleaved.push(app_r);
            interleaved.push(mic_l);
            interleaved.push(mic_r);
        }

        interleaved
    }

    /// Drain all remaining samples (for when stopping capture)
    /// This handles any leftover samples that couldn't be interleaved due to timing
    fn drain_remaining(&mut self) -> Vec<f32> {
        // If only one source, drain normally
        if self.has_app_audio && !self.has_mic_audio {
            return self.app_buffer.drain(..).collect();
        }
        if self.has_mic_audio && !self.has_app_audio {
            return self.mic_buffer.drain(..).collect();
        }

        // Both sources - force started flag and drain what we can
        self.started = true;
        let mut result = self.drain_interleaved();

        // Handle any remaining samples by padding with silence
        // This ensures we output complete 4-channel frames
        while !self.app_buffer.is_empty() || !self.mic_buffer.is_empty() {
            let app_l = self.app_buffer.pop_front().unwrap_or(0.0);
            let app_r = self.app_buffer.pop_front().unwrap_or(0.0);
            let mic_l = self.mic_buffer.pop_front().unwrap_or(0.0);
            let mic_r = self.mic_buffer.pop_front().unwrap_or(0.0);
            result.push(app_l);
            result.push(app_r);
            result.push(mic_l);
            result.push(mic_r);
        }

        result
    }
}

/// Capture mode for the session
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum CaptureMode {
    /// Capturing to ring buffer only (pre-recording)
    Monitoring = 0,
    /// Capturing to WAV file
    Recording = 1,
}

/// Manages an audio capture session on macOS
pub struct CaptureSession {
    config: Option<CaptureConfig>,
    monitor_config: Option<MonitorConfig>,
    command_tx: Option<Sender<CaptureCommand>>,
    thread_handle: Option<JoinHandle<()>>,
    is_monitoring: bool,
}

impl CaptureSession {
    pub fn new(config: CaptureConfig) -> Self {
        Self {
            config: Some(config),
            monitor_config: None,
            command_tx: None,
            thread_handle: None,
            is_monitoring: false,
        }
    }

    /// Create a session without an initial config (for monitoring-first usage)
    pub fn new_empty() -> Self {
        Self {
            config: None,
            monitor_config: None,
            command_tx: None,
            thread_handle: None,
            is_monitoring: false,
        }
    }

    /// Start capture in a background thread (direct recording mode)
    /// Returns a receiver for capture events
    pub fn start(&mut self) -> Result<Receiver<CaptureEvent>> {
        let config = self
            .config
            .clone()
            .ok_or_else(|| anyhow!("No capture config set"))?;

        let (event_tx, event_rx) = channel();
        let (command_tx, command_rx) = channel();

        let handle = thread::spawn(move || {
            capture_thread_main(config, command_rx, event_tx);
        });

        self.command_tx = Some(command_tx);
        self.thread_handle = Some(handle);
        self.is_monitoring = false;

        Ok(event_rx)
    }

    /// Start monitoring mode - captures audio to ring buffer without recording to file
    /// Returns a receiver for capture events (StatsUpdate with level meters)
    pub fn start_monitoring(&mut self, config: MonitorConfig) -> Result<Receiver<CaptureEvent>> {
        // Stop any existing session
        self.stop()?;

        let (event_tx, event_rx) = channel();
        let (command_tx, command_rx) = channel();

        let monitor_config = config.clone();
        self.monitor_config = Some(config);

        let handle = thread::spawn(move || {
            monitor_thread_main(monitor_config, command_rx, event_tx);
        });

        self.command_tx = Some(command_tx);
        self.thread_handle = Some(handle);
        self.is_monitoring = true;

        Ok(event_rx)
    }

    /// Transition from monitoring to recording mode
    /// The ring buffer contents are written to the beginning of the file
    pub fn start_recording(&mut self, output_path: PathBuf) -> Result<()> {
        if !self.is_monitoring {
            return Err(anyhow!("Not in monitoring mode"));
        }

        if let Some(tx) = &self.command_tx {
            tx.send(CaptureCommand::StartRecording { output_path })
                .map_err(|_| anyhow!("Failed to send start recording command"))?;
            self.is_monitoring = false;
            Ok(())
        } else {
            Err(anyhow!("No active monitoring session"))
        }
    }

    /// Resize the pre-roll buffer while monitoring
    pub fn resize_pre_roll(&mut self, duration_secs: f32) -> Result<()> {
        if !self.is_monitoring {
            return Err(anyhow!("Can only resize pre-roll while monitoring"));
        }

        if let Some(tx) = &self.command_tx {
            tx.send(CaptureCommand::ResizePreRoll { duration_secs })
                .map_err(|_| anyhow!("Failed to send resize command"))?;
            Ok(())
        } else {
            Err(anyhow!("No active monitoring session"))
        }
    }

    /// Stop the capture or monitoring
    pub fn stop(&mut self) -> Result<()> {
        if let Some(tx) = self.command_tx.take() {
            let _ = tx.send(CaptureCommand::Stop);
        }
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
        self.is_monitoring = false;
        Ok(())
    }

    /// Check if currently recording to file
    pub fn is_recording(&self) -> bool {
        self.command_tx.is_some() && !self.is_monitoring
    }

    /// Check if currently in monitoring mode
    pub fn is_monitoring(&self) -> bool {
        self.command_tx.is_some() && self.is_monitoring
    }
}

fn capture_thread_main(
    config: CaptureConfig,
    command_rx: Receiver<CaptureCommand>,
    event_tx: Sender<CaptureEvent>,
) {
    if let Err(e) = run_capture(&config, &command_rx, &event_tx) {
        let _ = event_tx.send(CaptureEvent::Error(e.to_string()));
    }

    let _ = event_tx.send(CaptureEvent::Stopped);
}

/// Shared state for the audio output handler (recording mode)
struct AudioOutputState {
    wav_writer: WavWriter<BufWriter<std::fs::File>>,
    total_frames: u64,
    total_bytes: u64,
    start_time: Instant,
    last_status_update: Instant,
    event_tx: Sender<CaptureEvent>,
    /// PIDs being captured (for per-source stats)
    pids: Vec<u32>,
    /// Channel buffer for interleaving app and mic audio to separate channels
    channel_buffer: AudioChannelBuffer,
    /// Number of output channels (2 for single source, 4 for both)
    output_channels: usize,
    /// Last RMS values for app audio (left, right)
    last_app_rms: (f32, f32),
    /// Last RMS values for mic audio (left, right)
    last_mic_rms: (f32, f32),
}

/// Shared state for the monitoring audio handler
struct MonitorOutputState {
    ring_buffer: AudioRingBuffer,
    start_time: Instant,
    last_status_update: Instant,
    event_tx: Sender<CaptureEvent>,
    /// Current capture mode
    mode: CaptureMode,
    /// WAV writer (only set after transitioning to recording)
    wav_writer: Option<WavWriter<BufWriter<std::fs::File>>>,
    /// Recording stats
    total_frames: u64,
    total_bytes: u64,
    recording_start_time: Option<Instant>,
    /// PIDs being captured (for per-source stats)
    pids: Vec<u32>,
    /// Channel buffer for interleaving app and mic audio to separate channels
    channel_buffer: AudioChannelBuffer,
    /// Number of output channels (2 for single source, 4 for both)
    output_channels: usize,
    /// Last RMS values for app audio (left, right)
    last_app_rms: (f32, f32),
    /// Last RMS values for mic audio (left, right)
    last_mic_rms: (f32, f32),
}

/// Handler for receiving audio samples from ScreenCaptureKit
struct AudioHandler {
    state: Arc<Mutex<AudioOutputState>>,
    stop_flag: Arc<AtomicBool>,
    /// Which audio source this handler is receiving
    audio_source: AudioSourceType,
}

impl SCStreamOutputTrait for AudioHandler {
    fn did_output_sample_buffer(&self, sample_buffer: CMSampleBuffer, of_type: SCStreamOutputType) {
        // Only accept the type we're registered for
        let expected_type = match self.audio_source {
            AudioSourceType::App => SCStreamOutputType::Audio,
            AudioSourceType::Microphone => SCStreamOutputType::Microphone,
        };
        if of_type != expected_type {
            return;
        }

        if self.stop_flag.load(Ordering::Relaxed) {
            return;
        }

        // Get audio buffer from sample buffer
        let Some(audio_buffers) = sample_buffer.audio_buffer_list() else {
            return;
        };

        let mut state = match self.state.lock() {
            Ok(s) => s,
            Err(_) => return,
        };

        // ScreenCaptureKit provides non-interleaved (planar) audio:
        // Each buffer contains samples for a single channel.
        // We need to interleave them for WAV output.
        if audio_buffers.num_buffers() == 0 {
            return;
        }

        // Get samples from each channel buffer
        let channel_samples: Vec<&[f32]> = audio_buffers
            .iter()
            .map(|b| bytes_to_f32_samples(b.data()))
            .collect();

        // All channels should have the same number of samples
        let num_frames = channel_samples.first().map(|s| s.len()).unwrap_or(0);
        if num_frames == 0 {
            return;
        }

        let num_channels = channel_samples.len();

        // Calculate RMS for this source before interleaving
        let left_rms_db = channel_samples
            .first()
            .map(|s| calculate_rms_db(s))
            .unwrap_or(-60.0);
        let right_rms_db = channel_samples
            .get(1)
            .map(|s| calculate_rms_db(s))
            .unwrap_or(-60.0);

        // Update RMS for the appropriate source
        match self.audio_source {
            AudioSourceType::App => {
                state.last_app_rms = (left_rms_db, right_rms_db);
            }
            AudioSourceType::Microphone => {
                state.last_mic_rms = (left_rms_db, right_rms_db);
            }
        }

        // Interleave samples: L0, R0, L1, R1, L2, R2, ...
        let mut interleaved = Vec::with_capacity(num_frames * num_channels);
        for frame_idx in 0..num_frames {
            for ch in 0..num_channels {
                if let Some(&sample) = channel_samples.get(ch).and_then(|s| s.get(frame_idx)) {
                    interleaved.push(sample);
                }
            }
        }

        // Push to channel buffer
        state.channel_buffer.push(self.audio_source, &interleaved);

        // Drain interleaved samples and write to WAV
        let output_samples = state.channel_buffer.drain_interleaved();
        let output_frames = output_samples.len() / state.output_channels;

        for &sample in &output_samples {
            if state.wav_writer.write_sample(sample).is_err() {
                return;
            }
        }

        if !output_samples.is_empty() {
            let num_bytes = output_samples.len() * 4; // f32 = 4 bytes
            state.total_frames += output_frames as u64;
            state.total_bytes += num_bytes as u64;
        }

        // Update status every 100ms
        if state.last_status_update.elapsed() >= Duration::from_millis(100) {
            let duration = state.start_time.elapsed();

            // Combined RMS: take the maximum of app and mic for each channel
            let combined_left = state.last_app_rms.0.max(state.last_mic_rms.0);
            let combined_right = state.last_app_rms.1.max(state.last_mic_rms.1);

            // Per-source stats: only available for single app capture
            // (multi-app capture produces pre-mixed audio from ScreenCaptureKit)
            let per_source_stats = if state.pids.len() == 1 {
                vec![SourceStats {
                    pid: state.pids[0],
                    left_rms_db: state.last_app_rms.0,
                    right_rms_db: state.last_app_rms.1,
                }]
            } else {
                Vec::new()
            };

            let stats = CaptureStats {
                duration_secs: duration.as_secs_f64(),
                total_frames: state.total_frames,
                file_size_bytes: state.total_bytes,
                buffer_frames: output_frames,
                is_recording: true,
                is_monitoring: false,
                pre_roll_buffer_secs: 0.0,
                left_rms_db: combined_left,
                right_rms_db: combined_right,
                per_source_stats,
            };
            let _ = state.event_tx.send(CaptureEvent::StatsUpdate(stats));
            state.last_status_update = Instant::now();
        }
    }
}

/// Handler for receiving audio samples during monitoring mode
struct MonitorAudioHandler {
    state: Arc<Mutex<MonitorOutputState>>,
    stop_flag: Arc<AtomicBool>,
    /// Which audio source this handler is receiving
    audio_source: AudioSourceType,
}

impl SCStreamOutputTrait for MonitorAudioHandler {
    fn did_output_sample_buffer(&self, sample_buffer: CMSampleBuffer, of_type: SCStreamOutputType) {
        // Only accept the type we're registered for
        let expected_type = match self.audio_source {
            AudioSourceType::App => SCStreamOutputType::Audio,
            AudioSourceType::Microphone => SCStreamOutputType::Microphone,
        };
        if of_type != expected_type {
            return;
        }

        if self.stop_flag.load(Ordering::Relaxed) {
            return;
        }

        // Get audio buffer from sample buffer
        let Some(audio_buffers) = sample_buffer.audio_buffer_list() else {
            return;
        };

        let mut state = match self.state.lock() {
            Ok(s) => s,
            Err(_) => return,
        };

        if audio_buffers.num_buffers() == 0 {
            return;
        }

        // Get samples from each channel buffer
        let channel_samples: Vec<&[f32]> = audio_buffers
            .iter()
            .map(|b| bytes_to_f32_samples(b.data()))
            .collect();

        let num_frames = channel_samples.first().map(|s| s.len()).unwrap_or(0);
        if num_frames == 0 {
            return;
        }

        let num_channels = channel_samples.len();

        // Calculate RMS for this source before interleaving
        let left_rms_db = channel_samples
            .first()
            .map(|s| calculate_rms_db(s))
            .unwrap_or(-60.0);
        let right_rms_db = channel_samples
            .get(1)
            .map(|s| calculate_rms_db(s))
            .unwrap_or(-60.0);

        // Update RMS for the appropriate source
        match self.audio_source {
            AudioSourceType::App => {
                state.last_app_rms = (left_rms_db, right_rms_db);
            }
            AudioSourceType::Microphone => {
                state.last_mic_rms = (left_rms_db, right_rms_db);
            }
        }

        // Interleave samples
        let mut interleaved = Vec::with_capacity(num_frames * num_channels);
        for frame_idx in 0..num_frames {
            for ch in 0..num_channels {
                if let Some(&sample) = channel_samples.get(ch).and_then(|s| s.get(frame_idx)) {
                    interleaved.push(sample);
                }
            }
        }

        // Push to channel buffer
        state.channel_buffer.push(self.audio_source, &interleaved);

        // Drain interleaved samples
        let output_samples = state.channel_buffer.drain_interleaved();
        let output_frames = output_samples.len() / state.output_channels;

        match state.mode {
            CaptureMode::Monitoring => {
                // Push interleaved samples to ring buffer
                if !output_samples.is_empty() {
                    state.ring_buffer.push(&output_samples);
                }
            }
            CaptureMode::Recording => {
                // Write interleaved samples to WAV file
                if let Some(ref mut wav_writer) = state.wav_writer {
                    for &sample in &output_samples {
                        if wav_writer.write_sample(sample).is_err() {
                            return;
                        }
                    }
                }
                if !output_samples.is_empty() {
                    let num_bytes = output_samples.len() * 4; // f32 = 4 bytes
                    state.total_frames += output_frames as u64;
                    state.total_bytes += num_bytes as u64;
                }
            }
        }

        // Update status every 100ms
        if state.last_status_update.elapsed() >= Duration::from_millis(100) {
            // Combined RMS: take the maximum of app and mic for each channel
            let combined_left = state.last_app_rms.0.max(state.last_mic_rms.0);
            let combined_right = state.last_app_rms.1.max(state.last_mic_rms.1);

            // Per-source stats: only available for single app capture
            // (multi-app capture produces pre-mixed audio from ScreenCaptureKit)
            let per_source_stats = if state.pids.len() == 1 {
                vec![SourceStats {
                    pid: state.pids[0],
                    left_rms_db: state.last_app_rms.0,
                    right_rms_db: state.last_app_rms.1,
                }]
            } else {
                Vec::new()
            };

            let stats = match state.mode {
                CaptureMode::Monitoring => CaptureStats {
                    duration_secs: state.start_time.elapsed().as_secs_f64(),
                    total_frames: 0,
                    file_size_bytes: 0,
                    buffer_frames: output_frames,
                    is_recording: false,
                    is_monitoring: true,
                    pre_roll_buffer_secs: state.ring_buffer.duration_secs(),
                    left_rms_db: combined_left,
                    right_rms_db: combined_right,
                    per_source_stats: per_source_stats.clone(),
                },
                CaptureMode::Recording => {
                    let duration = state
                        .recording_start_time
                        .map(|t| t.elapsed().as_secs_f64())
                        .unwrap_or(0.0);
                    CaptureStats {
                        duration_secs: duration,
                        total_frames: state.total_frames,
                        file_size_bytes: state.total_bytes,
                        buffer_frames: output_frames,
                        is_recording: true,
                        is_monitoring: false,
                        pre_roll_buffer_secs: 0.0,
                        left_rms_db: combined_left,
                        right_rms_db: combined_right,
                        per_source_stats,
                    }
                }
            };
            let _ = state.event_tx.send(CaptureEvent::StatsUpdate(stats));
            state.last_status_update = Instant::now();
        }
    }
}

/// Main entry point for monitoring thread
fn monitor_thread_main(
    config: MonitorConfig,
    command_rx: Receiver<CaptureCommand>,
    event_tx: Sender<CaptureEvent>,
) {
    if let Err(e) = run_monitor(&config, &command_rx, &event_tx) {
        let _ = event_tx.send(CaptureEvent::Error(e.to_string()));
    }

    let _ = event_tx.send(CaptureEvent::Stopped);
}

/// Run monitoring mode - captures to ring buffer, can transition to recording
fn run_monitor(
    config: &MonitorConfig,
    command_rx: &Receiver<CaptureCommand>,
    event_tx: &Sender<CaptureEvent>,
) -> Result<()> {
    // Validate that we have at least one source (PIDs or microphone)
    let has_apps = !config.pids.is_empty();
    let has_microphone = config.microphone_id.is_some();
    if !has_apps && !has_microphone {
        return Err(anyhow!("No audio source specified (no PIDs or microphone)"));
    }

    // Normalize sample rate to ScreenCaptureKit supported values (8000, 16000, 24000, 48000)
    let sample_rate = crate::types::normalize_sample_rate(config.sample_rate);
    if sample_rate != config.sample_rate {
        eprintln!(
            "Warning: Sample rate {} Hz is not supported by ScreenCaptureKit. Using {} Hz instead.",
            config.sample_rate, sample_rate
        );
    }

    // Validate that all processes exist (if any)
    for &pid in &config.pids {
        if !process_exists(pid) {
            return Err(anyhow!("Process with PID {} does not exist", pid));
        }
    }

    // Get shareable content
    let content = SCShareableContent::get()
        .map_err(|e| anyhow!("Failed to get shareable content: {:?}", e))?;

    // Find all applications by their PIDs (may be empty for mic-only capture)
    let all_apps = content.applications();
    let apps: Vec<_> = config
        .pids
        .iter()
        .filter_map(|&pid| {
            all_apps
                .iter()
                .find(|a| a.process_id() as u32 == pid)
                .cloned()
        })
        .collect();

    // For app capture, verify we found the apps
    if has_apps && apps.is_empty() {
        return Err(anyhow!(
            "No applications found in shareable content for the specified PIDs"
        ));
    }

    // Get the first display (needed for content filter)
    let displays = content.displays();
    let display = displays
        .first()
        .ok_or_else(|| anyhow!("No displays found"))?;

    // Create content filter based on capture mode
    let filter = if apps.is_empty() {
        // Microphone-only: use display filter (required by SCStream, but we won't capture app audio)
        SCContentFilter::builder().display(display).build()
    } else {
        // Collect windows from target apps
        let all_windows = content.windows();
        let app_pids: Vec<_> = apps.iter().map(|a| a.process_id()).collect();
        let target_windows: Vec<_> = all_windows
            .iter()
            .filter(|w| {
                w.owning_application()
                    .map(|app| app_pids.contains(&app.process_id()))
                    .unwrap_or(false)
            })
            .collect();

        if apps.len() == 1 {
            // For single app, try window-based capture first (more reliable)
            if let Some(window) = target_windows.first() {
                SCContentFilter::builder().window(window).build()
            } else {
                SCContentFilter::builder()
                    .display(display)
                    .include_applications(&[&apps[0]], &[])
                    .build()
            }
        } else {
            // For multiple apps, use display-level capture with application filter
            let app_refs: Vec<_> = apps.iter().collect();
            SCContentFilter::builder()
                .display(display)
                .include_applications(&app_refs, &[])
                .build()
        }
    };

    // Get display dimensions for stream config
    // Use small dimensions since we only care about audio
    let (width, height) = (100u32, 100u32);

    // Configure the stream for audio capture
    let mut stream_config = SCStreamConfiguration::new();
    stream_config
        .set_captures_audio(has_apps) // Only capture app audio if we have apps
        .set_sample_rate(sample_rate as i32)
        .set_channel_count(config.channels as i32)
        .set_width(width)
        .set_height(height);

    // Enable microphone capture if configured (macOS 15.0+)
    if let Some(ref mic_id) = config.microphone_id {
        stream_config
            .set_captures_microphone(true)
            .set_microphone_capture_device_id(mic_id);
    }

    // Calculate output channels: 4 if both sources active, 2 otherwise
    let output_channels: u16 = if has_apps && has_microphone {
        config.channels * 2 // 4 channels: app L/R + mic L/R
    } else {
        config.channels // 2 channels: single source
    };

    // Create ring buffer for pre-roll (sized for output channel count)
    let ring_buffer =
        AudioRingBuffer::new(config.pre_roll_duration_secs, sample_rate, output_channels);

    // WAV spec for when we transition to recording (uses output channel count)
    let wav_spec = WavSpec {
        channels: output_channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    // Create channel buffer for interleaving app and mic audio to separate channels
    let channel_buffer = AudioChannelBuffer::new(config.channels as usize, has_apps, has_microphone);

    // Create shared state for monitoring
    let state = Arc::new(Mutex::new(MonitorOutputState {
        ring_buffer,
        start_time: Instant::now(),
        last_status_update: Instant::now(),
        event_tx: event_tx.clone(),
        mode: CaptureMode::Monitoring,
        wav_writer: None,
        total_frames: 0,
        total_bytes: 0,
        recording_start_time: None,
        pids: config.pids.clone(),
        channel_buffer,
        output_channels: output_channels as usize,
        last_app_rms: (-60.0, -60.0),
        last_mic_rms: (-60.0, -60.0),
    }));

    let stop_flag = Arc::new(AtomicBool::new(false));

    // Create the stream with monitor handler
    let mut stream = SCStream::new(&filter, &stream_config);

    // Add handler for app audio if we have apps
    if has_apps {
        let audio_handler = MonitorAudioHandler {
            state: state.clone(),
            stop_flag: stop_flag.clone(),
            audio_source: AudioSourceType::App,
        };
        stream.add_output_handler(audio_handler, SCStreamOutputType::Audio);
    }

    // Add handler for microphone audio if configured
    if has_microphone {
        let mic_handler = MonitorAudioHandler {
            state: state.clone(),
            stop_flag: stop_flag.clone(),
            audio_source: AudioSourceType::Microphone,
        };
        stream.add_output_handler(mic_handler, SCStreamOutputType::Microphone);
    }

    // Start the stream
    stream
        .start_capture()
        .map_err(|e| anyhow!("Failed to start capture: {:?}", e))?;

    // Notify that monitoring has started
    let _ = event_tx.send(CaptureEvent::MonitoringStarted);

    // Process commands
    loop {
        match command_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(CaptureCommand::Stop) => break,
            Ok(CaptureCommand::StartRecording { output_path }) => {
                // Transition to recording mode
                let pre_roll_secs = {
                    let mut state = state.lock().map_err(|_| anyhow!("Failed to lock state"))?;

                    // Create WAV file
                    let file = std::fs::File::create(&output_path).with_context(|| {
                        format!("Failed to create output file: {}", output_path.display())
                    })?;
                    let buf_writer = BufWriter::new(file);
                    let mut wav_writer = WavWriter::new(buf_writer, wav_spec)
                        .context("Failed to create WAV writer")?;

                    // Drain ring buffer and write to WAV
                    let pre_roll_secs = state.ring_buffer.duration_secs();
                    let samples = state.ring_buffer.drain();
                    for sample in samples {
                        wav_writer.write_sample(sample)?;
                    }

                    // Update state
                    state.wav_writer = Some(wav_writer);
                    state.mode = CaptureMode::Recording;
                    state.recording_start_time = Some(Instant::now());
                    state.total_frames = 0;
                    state.total_bytes = 0;

                    pre_roll_secs
                };

                let _ = event_tx.send(CaptureEvent::RecordingStarted { pre_roll_secs });
            }
            Ok(CaptureCommand::ResizePreRoll { duration_secs }) => {
                let mut state = state.lock().map_err(|_| anyhow!("Failed to lock state"))?;
                state.ring_buffer.resize(duration_secs);
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    // Signal stop to the output handler
    stop_flag.store(true, Ordering::Relaxed);

    // Stop the stream
    stream
        .stop_capture()
        .map_err(|e| anyhow!("Failed to stop capture: {:?}", e))?;

    // Finalize WAV file if we were recording
    if let Ok(mut state) = state.lock() {
        // Drain any remaining samples from the channel buffer
        let remaining = state.channel_buffer.drain_remaining();

        if let Some(ref mut wav_writer) = state.wav_writer {
            for &sample in &remaining {
                let _ = wav_writer.write_sample(sample);
            }
        }

        if let Some(wav_writer) = state.wav_writer.take() {
            let _ = wav_writer.finalize();
        }
    }

    Ok(())
}

fn run_capture(
    config: &CaptureConfig,
    command_rx: &Receiver<CaptureCommand>,
    event_tx: &Sender<CaptureEvent>,
) -> Result<()> {
    // Validate that we have at least one audio source (PIDs or microphone)
    let has_apps = !config.pids.is_empty();
    let has_microphone = config.microphone_id.is_some();
    if !has_apps && !has_microphone {
        return Err(anyhow!("No audio source specified (no PIDs or microphone)"));
    }

    // Normalize sample rate to ScreenCaptureKit supported values (8000, 16000, 24000, 48000)
    let sample_rate = crate::types::normalize_sample_rate(config.sample_rate);
    if sample_rate != config.sample_rate {
        eprintln!(
            "Warning: Sample rate {} Hz is not supported by ScreenCaptureKit. Using {} Hz instead.",
            config.sample_rate, sample_rate
        );
    }

    // Validate that all processes exist (only if we have apps to capture)
    if has_apps {
        for &pid in &config.pids {
            if !process_exists(pid) {
                return Err(anyhow!("Process with PID {} does not exist", pid));
            }
        }
    }

    // Get shareable content
    let content = SCShareableContent::get()
        .map_err(|e| anyhow!("Failed to get shareable content: {:?}", e))?;

    // Find all applications by their PIDs
    let all_apps = content.applications();
    let apps: Vec<_> = config
        .pids
        .iter()
        .filter_map(|&pid| {
            all_apps
                .iter()
                .find(|a| a.process_id() as u32 == pid)
                .cloned()
        })
        .collect();

    // Only require apps if we're capturing app audio (not mic-only)
    if has_apps && apps.is_empty() {
        return Err(anyhow!(
            "No applications found in shareable content for the specified PIDs"
        ));
    }

    // Get the first display
    let displays = content.displays();
    let display = displays
        .first()
        .ok_or_else(|| anyhow!("No displays found"))?;

    // Collect windows from target apps
    let all_windows = content.windows();
    let app_pids: Vec<_> = apps.iter().map(|a| a.process_id()).collect();
    let target_windows: Vec<_> = all_windows
        .iter()
        .filter(|w| {
            w.owning_application()
                .map(|app| app_pids.contains(&app.process_id()))
                .unwrap_or(false)
        })
        .collect();

    // Create content filter
    let filter = if apps.is_empty() {
        // Microphone-only capture: use display filter without app audio
        SCContentFilter::builder().display(display).build()
    } else if apps.len() == 1 {
        // For single app, try window-based capture first (more reliable)
        if let Some(window) = target_windows.first() {
            SCContentFilter::builder().window(window).build()
        } else {
            SCContentFilter::builder()
                .display(display)
                .include_applications(&[&apps[0]], &[])
                .build()
        }
    } else {
        // For multiple apps, use display-level capture with application filter
        let app_refs: Vec<_> = apps.iter().collect();
        SCContentFilter::builder()
            .display(display)
            .include_applications(&app_refs, &[])
            .build()
    };

    // Get display dimensions for stream config
    // Display-level capture may require valid video dimensions even for audio-only
    let (width, height) = if apps.len() == 1 && target_windows.first().is_some() {
        // For single window capture, minimal dimensions work
        (1u32, 1u32)
    } else {
        // For display-level capture, use small but valid dimensions
        (100u32, 100u32)
    };

    // Configure the stream for audio capture
    let mut stream_config = SCStreamConfiguration::new();
    stream_config
        .set_captures_audio(has_apps) // Only capture app audio if we have apps
        .set_sample_rate(sample_rate as i32)
        .set_channel_count(config.channels as i32)
        .set_width(width)
        .set_height(height);

    // Enable microphone capture if configured (macOS 15.0+)
    if let Some(ref mic_id) = config.microphone_id {
        stream_config
            .set_captures_microphone(true)
            .set_microphone_capture_device_id(mic_id);
    }

    // Calculate output channels: 4 if both sources active, 2 otherwise
    let output_channels: u16 = if has_apps && has_microphone {
        config.channels * 2 // 4 channels: app L/R + mic L/R
    } else {
        config.channels // 2 channels: single source
    };

    // Set up WAV writer with output channel count
    let wav_spec = WavSpec {
        channels: output_channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let file = std::fs::File::create(&config.output_path).with_context(|| {
        format!(
            "Failed to create output file: {}",
            config.output_path.display()
        )
    })?;
    let buf_writer = BufWriter::new(file);
    let wav_writer = WavWriter::new(buf_writer, wav_spec).context("Failed to create WAV writer")?;

    // Create channel buffer for interleaving app and mic audio to separate channels
    let channel_buffer = AudioChannelBuffer::new(config.channels as usize, has_apps, has_microphone);

    // Create shared state for the output handler
    let state = Arc::new(Mutex::new(AudioOutputState {
        wav_writer,
        total_frames: 0,
        total_bytes: 0,
        start_time: Instant::now(),
        last_status_update: Instant::now(),
        event_tx: event_tx.clone(),
        pids: config.pids.clone(),
        channel_buffer,
        output_channels: output_channels as usize,
        last_app_rms: (-60.0, -60.0),
        last_mic_rms: (-60.0, -60.0),
    }));

    let stop_flag = Arc::new(AtomicBool::new(false));

    // Create the stream with output handler
    let mut stream = SCStream::new(&filter, &stream_config);

    // Add handler for app audio if we have apps
    if has_apps {
        let audio_handler = AudioHandler {
            state: state.clone(),
            stop_flag: stop_flag.clone(),
            audio_source: AudioSourceType::App,
        };
        stream.add_output_handler(audio_handler, SCStreamOutputType::Audio);
    }

    // Add handler for microphone audio if configured
    if has_microphone {
        let mic_handler = AudioHandler {
            state: state.clone(),
            stop_flag: stop_flag.clone(),
            audio_source: AudioSourceType::Microphone,
        };
        stream.add_output_handler(mic_handler, SCStreamOutputType::Microphone);
    }

    // Start the stream
    stream
        .start_capture()
        .map_err(|e| anyhow!("Failed to start capture: {:?}", e))?;

    // Notify that capture has started
    let _ = event_tx.send(CaptureEvent::Started {
        buffer_size: config.sample_rate as usize / 10, // Approximate 100ms buffer
    });

    // Wait for stop command
    loop {
        match command_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(CaptureCommand::Stop) => break,
            Ok(CaptureCommand::StartRecording { .. }) => {
                // Ignored in direct capture mode - already recording
                continue;
            }
            Ok(CaptureCommand::ResizePreRoll { .. }) => {
                // Ignored in direct capture mode - no ring buffer
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    // Signal stop to the output handler
    stop_flag.store(true, Ordering::Relaxed);

    // Stop the stream
    stream
        .stop_capture()
        .map_err(|e| anyhow!("Failed to stop capture: {:?}", e))?;

    // Finalize the WAV file
    if let Ok(mut state) = state.lock() {
        // Drain any remaining samples from the channel buffer
        let remaining = state.channel_buffer.drain_remaining();
        for &sample in &remaining {
            let _ = state.wav_writer.write_sample(sample);
        }

        // Flush and finalize happens when wav_writer is dropped
        // We need to take ownership to finalize
        let _ = std::mem::replace(
            &mut state.wav_writer,
            WavWriter::new(
                BufWriter::new(std::fs::File::create("/dev/null").unwrap()),
                wav_spec,
            )
            .unwrap(),
        )
        .finalize();
    }

    Ok(())
}

/// Safely cast a byte slice to a slice of f32
fn bytes_to_f32_samples(bytes: &[u8]) -> &[f32] {
    let len = bytes.len() / 4;
    if len == 0 || bytes.len() % 4 != 0 {
        return &[];
    }
    if bytes.as_ptr() as usize % std::mem::align_of::<f32>() != 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len) }
}

/// Calculate RMS level in dB from audio samples
/// Returns -60.0 dB for silence/zero signal
fn calculate_rms_db(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return -60.0;
    }
    let sum_squares: f32 = samples.iter().map(|&s| s * s).sum();
    let rms = (sum_squares / samples.len() as f32).sqrt();
    if rms <= 0.0 {
        -60.0
    } else {
        20.0 * rms.log10()
    }
}
