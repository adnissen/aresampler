//! macOS audio capture implementation using ScreenCaptureKit

use crate::core::process::process_exists;
use crate::core::ring_buffer::AudioRingBuffer;
use crate::core::types::{
    CaptureCommand, CaptureConfig, CaptureEvent, CaptureStats, MonitorConfig, SourceStats,
};
use anyhow::{Context, Result, anyhow};
use hound::{SampleFormat, WavSpec, WavWriter};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use screencapturekit::cm::CMSampleBuffer;
use screencapturekit::shareable_content::SCShareableContent;
use screencapturekit::stream::SCStream;
use screencapturekit::stream::configuration::SCStreamConfiguration;
use screencapturekit::stream::content_filter::SCContentFilter;
use screencapturekit::stream::output_trait::SCStreamOutputTrait;
use screencapturekit::stream::output_type::SCStreamOutputType;
use std::collections::VecDeque;
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Audio source type for channel routing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AudioSourceType {
    App,
    Microphone,
}

/// Shared buffer for microphone audio samples.
/// The callback pushes samples here; a separate thread reads and resamples them.
struct MicrophoneBuffer {
    /// Raw interleaved samples from the microphone at its native rate
    samples: VecDeque<f32>,
    /// Input sample rate detected from the audio format
    input_sample_rate: u32,
    /// Number of channels in the buffer
    channels: usize,
    /// Left channel RMS in dB (for level meters)
    left_rms_db: f32,
    /// Right channel RMS in dB (for level meters)
    right_rms_db: f32,
}

impl MicrophoneBuffer {
    fn new(channels: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(48000 * channels), // ~1 second buffer
            input_sample_rate: 48000, // Default until we detect actual rate
            channels,
            left_rms_db: -60.0,
            right_rms_db: -60.0,
        }
    }
}

/// Lightweight handler that only buffers microphone samples.
/// No resampling is done here - that's handled by a separate processing thread.
struct MicrophoneBufferHandler {
    /// Shared buffer where samples are pushed
    buffer: Arc<Mutex<MicrophoneBuffer>>,
    /// Stop flag for early termination
    stop_flag: Arc<AtomicBool>,
    /// Target channel count (typically 2 for stereo)
    target_channels: usize,
}

impl SCStreamOutputTrait for MicrophoneBufferHandler {
    fn did_output_sample_buffer(&self, sample_buffer: CMSampleBuffer, of_type: SCStreamOutputType) {
        // Only accept microphone audio
        if of_type != SCStreamOutputType::Microphone {
            return;
        }

        if self.stop_flag.load(Ordering::Relaxed) {
            return;
        }

        // Get audio buffer from sample buffer
        let Some(audio_buffers) = sample_buffer.audio_buffer_list() else {
            return;
        };

        if audio_buffers.num_buffers() == 0 {
            return;
        }

        // Detect input sample rate from the format description
        let input_sample_rate = sample_buffer
            .format_description()
            .and_then(|fd| fd.audio_sample_rate())
            .map(|r| r as u32)
            .unwrap_or(48000);

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

        // Calculate RMS for level meters
        let left_rms_db = channel_samples
            .first()
            .map(|s| calculate_rms_db(s))
            .unwrap_or(-60.0);
        let right_rms_db = channel_samples
            .get(1)
            .map(|s| calculate_rms_db(s))
            .unwrap_or(left_rms_db); // Use left if mono

        // Interleave samples, converting mono to stereo if needed
        let interleaved = if num_channels == 1 && self.target_channels == 2 {
            // Mono to stereo: duplicate each sample to both L and R
            let mono_samples = channel_samples.first().unwrap();
            let mut stereo = Vec::with_capacity(num_frames * 2);
            for &sample in mono_samples.iter() {
                stereo.push(sample); // L
                stereo.push(sample); // R
            }
            stereo
        } else {
            // Normal interleaving
            let mut interleaved = Vec::with_capacity(num_frames * num_channels);
            for frame_idx in 0..num_frames {
                for ch in 0..num_channels {
                    if let Some(&sample) = channel_samples.get(ch).and_then(|s| s.get(frame_idx)) {
                        interleaved.push(sample);
                    }
                }
            }
            interleaved
        };

        // Push to shared buffer - NO RESAMPLING HERE
        if let Ok(mut buf) = self.buffer.lock() {
            buf.samples.extend(interleaved);
            buf.input_sample_rate = input_sample_rate;
            buf.left_rms_db = left_rms_db;
            buf.right_rms_db = right_rms_db;
        }
    }
}

/// Message sent from mic processing thread to main loop
struct MicProcessedAudio {
    /// Resampled interleaved samples at target rate
    samples: Vec<f32>,
    /// Source index in the channel buffer
    source_index: usize,
    /// RMS levels for this chunk
    left_rms_db: f32,
    right_rms_db: f32,
}

/// Processing loop that reads from microphone buffer, resamples, and sends to channel.
/// This runs in a separate thread to avoid blocking the audio callback.
fn mic_processing_loop(
    buffer: Arc<Mutex<MicrophoneBuffer>>,
    stop_flag: Arc<AtomicBool>,
    target_sample_rate: u32,
    channels: usize,
    source_index: usize,
    output_tx: Sender<MicProcessedAudio>,
) {
    // Fixed chunk size for consistent rubato behavior
    let chunk_size = 1024;

    // Will be initialized when we know input rate
    let mut resampler: Option<SincFixedIn<f32>> = None;
    let mut current_input_rate: Option<u32> = None;

    loop {
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }

        // Get input rate, RMS, and check if we have enough samples
        let (input_rate, has_enough, left_rms, right_rms) = {
            let buf = buffer.lock().unwrap();
            let sample_count = chunk_size * channels;
            (
                buf.input_sample_rate,
                buf.samples.len() >= sample_count,
                buf.left_rms_db,
                buf.right_rms_db,
            )
        };

        if !has_enough {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        }

        // Initialize or reinitialize resampler if input rate changed
        if current_input_rate != Some(input_rate) {
            if input_rate != target_sample_rate {
                let ratio = target_sample_rate as f64 / input_rate as f64;
                let params = SincInterpolationParameters {
                    sinc_len: 256,
                    f_cutoff: 0.95,
                    interpolation: SincInterpolationType::Linear,
                    oversampling_factor: 256,
                    window: WindowFunction::BlackmanHarris2,
                };

                match SincFixedIn::new(ratio, 2.0, params, chunk_size, channels) {
                    Ok(r) => {
                        eprintln!(
                            "Mic resampler initialized: {} Hz -> {} Hz (ratio: {:.4})",
                            input_rate, target_sample_rate, ratio
                        );
                        resampler = Some(r);
                    }
                    Err(e) => {
                        eprintln!("Failed to create mic resampler: {:?}", e);
                        resampler = None;
                    }
                }
            } else {
                resampler = None; // No resampling needed
            }
            current_input_rate = Some(input_rate);
        }

        // Drain exactly chunk_size frames from buffer
        let samples: Vec<f32> = {
            let mut buf = buffer.lock().unwrap();
            buf.samples.drain(..chunk_size * channels).collect()
        };

        // De-interleave for rubato (needs planar format)
        let mut planar: Vec<Vec<f32>> = vec![vec![0.0; chunk_size]; channels];
        for (i, &s) in samples.iter().enumerate() {
            let ch = i % channels;
            let frame = i / channels;
            if frame < chunk_size {
                planar[ch][frame] = s;
            }
        }

        // Resample (or pass through if rates match)
        let output = if let Some(ref mut r) = resampler {
            match r.process(&planar, None) {
                Ok(resampled) => {
                    // Re-interleave the output
                    let output_frames = resampled.first().map(|v| v.len()).unwrap_or(0);
                    let mut interleaved = Vec::with_capacity(output_frames * channels);
                    for frame_idx in 0..output_frames {
                        for ch in 0..channels {
                            interleaved.push(resampled[ch].get(frame_idx).copied().unwrap_or(0.0));
                        }
                    }
                    interleaved
                }
                Err(e) => {
                    eprintln!("Resampling error: {:?}", e);
                    samples // Fallback to original
                }
            }
        } else {
            samples // No resampling needed
        };

        // Send to main loop
        let _ = output_tx.send(MicProcessedAudio {
            samples: output,
            source_index,
            left_rms_db: left_rms,
            right_rms_db: right_rms,
        });
    }
}

/// Channel buffer that accumulates samples from multiple sources and interleaves them.
///
/// ## Output Channel Layout
///
/// The output WAV file uses a multi-channel format where each audio source occupies
/// its own stereo pair:
///
/// - Source 0 (apps, if enabled): channels 0-1
/// - Source 1 (first mic): channels 2-3
/// - Source 2 (second mic): channels 4-5
/// - ... and so on
///
/// Each audio frame contains one sample per channel in order:
/// `[ch0, ch1, ch2, ch3, ...]` repeated for each sample frame.
///
/// This format allows DAWs to easily split sources into separate tracks.
struct AudioChannelBuffer {
    /// Per-source audio buffers (each contains interleaved stereo: L, R, L, R, ...)
    source_buffers: Vec<VecDeque<f32>>,
    /// Number of input channels per source (typically 2 for stereo)
    input_channels: usize,
    /// Number of output channels (input_channels * source_count)
    output_channels: usize,
    /// Number of active sources
    source_count: usize,
    /// Whether we've started outputting (after initial sync)
    started: bool,
    /// Minimum frames to buffer before starting (for initial sync when multiple sources)
    min_buffer_frames: usize,
}

impl AudioChannelBuffer {
    /// Create a new channel buffer for the specified number of sources.
    ///
    /// # Arguments
    /// * `source_count` - Number of audio sources (apps count as 1 source, each mic is 1 source)
    /// * `input_channels` - Channels per source (typically 2 for stereo)
    fn new(source_count: usize, input_channels: usize) -> Self {
        // Output channels: input_channels per source
        let output_channels = source_count * input_channels;

        // Buffer ~50ms of audio at 48kHz before starting output (for sync)
        // This gives all sources time to start delivering
        let min_buffer_frames = if source_count > 1 {
            2400 // ~50ms at 48kHz
        } else {
            0 // No buffering needed for single source
        };

        Self {
            source_buffers: (0..source_count).map(|_| VecDeque::new()).collect(),
            input_channels,
            output_channels,
            source_count,
            started: false,
            min_buffer_frames,
        }
    }

    /// Push samples from a specific source by index.
    ///
    /// # Arguments
    /// * `source_index` - Index of the source (0 = apps if enabled, 1+ = microphones)
    /// * `samples` - Interleaved stereo samples
    fn push(&mut self, source_index: usize, samples: &[f32]) {
        if let Some(buffer) = self.source_buffers.get_mut(source_index) {
            buffer.extend(samples);
        }
    }

    /// Drain and interleave available samples from all active sources.
    ///
    /// Returns interleaved samples: `[src0_L, src0_R, src1_L, src1_R, ...]` per frame
    fn drain_interleaved(&mut self) -> Vec<f32> {
        // Single source - just drain directly
        if self.source_count == 1 {
            if let Some(buffer) = self.source_buffers.get_mut(0) {
                return buffer.drain(..).collect();
            }
            return Vec::new();
        }

        // Multiple sources - find minimum available frames across all sources
        let frame_counts: Vec<usize> = self
            .source_buffers
            .iter()
            .map(|b| b.len() / self.input_channels)
            .collect();

        // Wait for all sources to have minimum buffer before starting
        if !self.started {
            if frame_counts.iter().all(|&f| f >= self.min_buffer_frames) {
                self.started = true;
            } else {
                return Vec::new();
            }
        }

        // Once started, output the minimum of all sources to stay synchronized
        let available_frames = frame_counts.iter().copied().min().unwrap_or(0);

        if available_frames == 0 {
            return Vec::new();
        }

        // Interleave: output_channels samples per frame
        let mut interleaved = Vec::with_capacity(available_frames * self.output_channels);

        for _ in 0..available_frames {
            // Pop one frame from each source and interleave
            for buffer in &mut self.source_buffers {
                for _ in 0..self.input_channels {
                    interleaved.push(buffer.pop_front().unwrap_or(0.0));
                }
            }
        }

        interleaved
    }

    /// Drain all remaining samples (for when stopping capture).
    ///
    /// This handles any leftover samples that couldn't be interleaved due to timing,
    /// padding with silence to ensure complete multi-channel frames.
    fn drain_remaining(&mut self) -> Vec<f32> {
        // Single source - drain normally
        if self.source_count == 1 {
            if let Some(buffer) = self.source_buffers.get_mut(0) {
                return buffer.drain(..).collect();
            }
            return Vec::new();
        }

        // Multiple sources - force started and drain what we can
        self.started = true;
        let mut result = self.drain_interleaved();

        // Handle any remaining samples by padding with silence
        // This ensures we output complete multi-channel frames
        while self.source_buffers.iter().any(|b| !b.is_empty()) {
            for buffer in &mut self.source_buffers {
                for _ in 0..self.input_channels {
                    result.push(buffer.pop_front().unwrap_or(0.0));
                }
            }
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
    /// Channel buffer for interleaving audio from multiple sources
    channel_buffer: AudioChannelBuffer,
    /// Number of output channels
    output_channels: usize,
    /// Last RMS values per source (each entry is (left_db, right_db))
    /// Index 0 = apps (if enabled), subsequent indices = microphones
    source_rms: Vec<(f32, f32)>,
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
    /// Channel buffer for interleaving audio from multiple sources
    channel_buffer: AudioChannelBuffer,
    /// Number of output channels
    output_channels: usize,
    /// Last RMS values per source (each entry is (left_db, right_db))
    /// Index 0 = apps (if enabled), subsequent indices = microphones
    source_rms: Vec<(f32, f32)>,
    /// Target channel count per source (typically 2 for stereo)
    target_channels: usize,
}

/// Handler for receiving audio samples from ScreenCaptureKit (app audio)
struct AudioHandler {
    state: Arc<Mutex<AudioOutputState>>,
    stop_flag: Arc<AtomicBool>,
    /// Index of this source in the channel buffer (0 = apps)
    source_index: usize,
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

        // Update RMS for this source index
        if let Some(rms) = state.source_rms.get_mut(self.source_index) {
            *rms = (left_rms_db, right_rms_db);
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

        // Push to channel buffer using source index
        state.channel_buffer.push(self.source_index, &interleaved);

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

            // Combined RMS: take the maximum across all sources for each channel
            let combined_left = state
                .source_rms
                .iter()
                .map(|(l, _)| *l)
                .fold(-60.0_f32, f32::max);
            let combined_right = state
                .source_rms
                .iter()
                .map(|(_, r)| *r)
                .fold(-60.0_f32, f32::max);

            // Per-source stats: only available for single app capture
            // (multi-app capture produces pre-mixed audio from ScreenCaptureKit)
            let per_source_stats = if state.pids.len() == 1 && !state.source_rms.is_empty() {
                vec![SourceStats {
                    pid: state.pids[0],
                    left_rms_db: state.source_rms[0].0,
                    right_rms_db: state.source_rms[0].1,
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
    /// Index of this source in the channel buffer
    source_index: usize,
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

        // Update RMS for this source index
        if let Some(rms) = state.source_rms.get_mut(self.source_index) {
            *rms = (left_rms_db, right_rms_db);
        }

        // Interleave samples, converting mono to stereo if needed
        let target_channels = state.target_channels;
        let interleaved = if num_channels == 1 && target_channels == 2 {
            // Mono to stereo: duplicate each sample to both L and R
            let mono_samples = channel_samples.first().unwrap();
            let mut stereo = Vec::with_capacity(num_frames * 2);
            for &sample in mono_samples.iter() {
                stereo.push(sample); // L
                stereo.push(sample); // R
            }
            stereo
        } else {
            // Normal interleaving for stereo or matching channel counts
            let mut interleaved = Vec::with_capacity(num_frames * num_channels);
            for frame_idx in 0..num_frames {
                for ch in 0..num_channels {
                    if let Some(&sample) = channel_samples.get(ch).and_then(|s| s.get(frame_idx)) {
                        interleaved.push(sample);
                    }
                }
            }
            interleaved
        };

        // Push to channel buffer using source index
        state.channel_buffer.push(self.source_index, &interleaved);

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
            // Combined RMS: take the maximum across all sources for each channel
            let combined_left = state
                .source_rms
                .iter()
                .map(|(l, _)| *l)
                .fold(-60.0_f32, f32::max);
            let combined_right = state
                .source_rms
                .iter()
                .map(|(_, r)| *r)
                .fold(-60.0_f32, f32::max);

            // Per-source stats: only available for single app capture
            // (multi-app capture produces pre-mixed audio from ScreenCaptureKit)
            let per_source_stats = if state.pids.len() == 1 && !state.source_rms.is_empty() {
                vec![SourceStats {
                    pid: state.pids[0],
                    left_rms_db: state.source_rms[0].0,
                    right_rms_db: state.source_rms[0].1,
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
    // Validate that we have at least one source (PIDs or microphones)
    let has_apps = !config.pids.is_empty();
    let mic_count = config.microphones.len();
    if !has_apps && mic_count == 0 {
        return Err(anyhow!(
            "No audio source specified (no PIDs or microphones)"
        ));
    }

    // Normalize sample rate to ScreenCaptureKit supported values (8000, 16000, 24000, 48000)
    let sample_rate = crate::core::types::normalize_sample_rate(config.sample_rate);
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

    // Create content filter for app stream
    let app_filter = if apps.is_empty() {
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

    // Calculate source count: 1 for apps (if any) + N microphones
    let source_count = if has_apps { 1 } else { 0 } + mic_count;

    // Calculate output channels: 2 per source
    let output_channels: u16 = (source_count * config.channels as usize) as u16;

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

    // Create channel buffer for interleaving audio from multiple sources
    let channel_buffer = AudioChannelBuffer::new(source_count, config.channels as usize);

    // Initialize RMS tracking for each source
    let source_rms: Vec<(f32, f32)> = (0..source_count).map(|_| (-60.0, -60.0)).collect();

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
        source_rms,
        target_channels: config.channels as usize,
    }));

    let stop_flag = Arc::new(AtomicBool::new(false));

    // Track all streams we create
    let mut streams: Vec<SCStream> = Vec::new();

    // Create the app stream (if we have apps or need a base stream for mic-only)
    if has_apps || mic_count == 0 {
        let mut app_stream_config = SCStreamConfiguration::new();
        app_stream_config
            .set_captures_audio(has_apps) // Only capture app audio if we have apps
            .set_sample_rate(sample_rate as i32)
            .set_channel_count(config.channels as i32)
            .set_width(width)
            .set_height(height);
        // Note: No microphone config on app stream - mics get their own streams

        let mut app_stream = SCStream::new(&app_filter, &app_stream_config);

        // Add handler for app audio
        if has_apps {
            let audio_handler = MonitorAudioHandler {
                state: state.clone(),
                stop_flag: stop_flag.clone(),
                source_index: 0, // Apps are always source index 0
                audio_source: AudioSourceType::App,
            };
            app_stream.add_output_handler(audio_handler, SCStreamOutputType::Audio);
        }

        streams.push(app_stream);
    }

    // Create channel for receiving processed mic audio
    let (mic_tx, mic_rx) = channel::<MicProcessedAudio>();

    // Track mic processing threads
    let mut mic_threads: Vec<JoinHandle<()>> = Vec::new();

    // Create separate streams for each microphone
    for (mic_idx, mic_config) in config.microphones.iter().enumerate() {
        // Source index: 0 if no apps, otherwise 1+
        let source_index = if has_apps { 1 + mic_idx } else { mic_idx };

        // Create shared buffer for this microphone
        let mic_buffer = Arc::new(Mutex::new(MicrophoneBuffer::new(config.channels as usize)));

        // Create minimal content filter for mic-only stream
        let mic_filter = SCContentFilter::builder().display(display).build();

        // Configure for microphone only - don't set sample rate to use native
        let mut mic_stream_config = SCStreamConfiguration::new();
        mic_stream_config
            .set_captures_audio(false) // No app audio
            .set_captures_microphone(true)
            .set_microphone_capture_device_id(&mic_config.id)
            .set_channel_count(config.channels as i32)
            .set_width(width)
            .set_height(height);
        // Note: Don't set sample_rate - let it use native rate for resampling

        let mut mic_stream = SCStream::new(&mic_filter, &mic_stream_config);

        // Add lightweight buffer handler (no resampling in callback)
        let mic_handler = MicrophoneBufferHandler {
            buffer: mic_buffer.clone(),
            stop_flag: stop_flag.clone(),
            target_channels: config.channels as usize,
        };
        mic_stream.add_output_handler(mic_handler, SCStreamOutputType::Microphone);

        streams.push(mic_stream);

        // Spawn processing thread for this microphone
        let buffer_clone = mic_buffer.clone();
        let stop_flag_clone = stop_flag.clone();
        let mic_tx_clone = mic_tx.clone();
        let channels = config.channels as usize;

        let thread = thread::spawn(move || {
            mic_processing_loop(
                buffer_clone,
                stop_flag_clone,
                sample_rate,
                channels,
                source_index,
                mic_tx_clone,
            );
        });
        mic_threads.push(thread);
    }

    // Drop the original sender so mic_rx will close when all threads are done
    drop(mic_tx);

    // Start all streams
    for stream in &mut streams {
        stream
            .start_capture()
            .map_err(|e| anyhow!("Failed to start capture: {:?}", e))?;
    }

    // Notify that monitoring has started
    let _ = event_tx.send(CaptureEvent::MonitoringStarted);

    // Process commands and mic audio
    loop {
        // Process any pending mic audio (non-blocking)
        while let Ok(mic_audio) = mic_rx.try_recv() {
            if let Ok(mut state_guard) = state.lock() {
                // Update RMS for this source
                if let Some(rms) = state_guard.source_rms.get_mut(mic_audio.source_index) {
                    *rms = (mic_audio.left_rms_db, mic_audio.right_rms_db);
                }

                // Push to channel buffer
                state_guard
                    .channel_buffer
                    .push(mic_audio.source_index, &mic_audio.samples);

                // Drain and write/buffer
                let output_samples = state_guard.channel_buffer.drain_interleaved();
                let output_frames = output_samples.len() / state_guard.output_channels;

                match state_guard.mode {
                    CaptureMode::Monitoring => {
                        if !output_samples.is_empty() {
                            state_guard.ring_buffer.push(&output_samples);
                        }
                    }
                    CaptureMode::Recording => {
                        if let Some(ref mut wav_writer) = state_guard.wav_writer {
                            for &sample in &output_samples {
                                let _ = wav_writer.write_sample(sample);
                            }
                        }
                        if !output_samples.is_empty() {
                            let num_bytes = output_samples.len() * 4;
                            state_guard.total_frames += output_frames as u64;
                            state_guard.total_bytes += num_bytes as u64;
                        }
                    }
                }

                // Send stats update if enough time has passed (100ms)
                // This is needed when there's no app audio handler to send updates
                if state_guard.last_status_update.elapsed() >= Duration::from_millis(100) {
                    // Combined RMS from all sources
                    let combined_left = state_guard
                        .source_rms
                        .iter()
                        .map(|(l, _)| *l)
                        .fold(-60.0_f32, f32::max);
                    let combined_right = state_guard
                        .source_rms
                        .iter()
                        .map(|(_, r)| *r)
                        .fold(-60.0_f32, f32::max);

                    let stats = match state_guard.mode {
                        CaptureMode::Monitoring => CaptureStats {
                            duration_secs: state_guard.start_time.elapsed().as_secs_f64(),
                            total_frames: state_guard.total_frames,
                            file_size_bytes: state_guard.total_bytes,
                            buffer_frames: output_frames,
                            is_recording: false,
                            is_monitoring: true,
                            pre_roll_buffer_secs: state_guard.ring_buffer.duration_secs(),
                            left_rms_db: combined_left,
                            right_rms_db: combined_right,
                            per_source_stats: Vec::new(),
                        },
                        CaptureMode::Recording => {
                            // Use recording_start_time so duration resets when record is pressed
                            let duration = state_guard
                                .recording_start_time
                                .map(|t| t.elapsed().as_secs_f64())
                                .unwrap_or(0.0);
                            CaptureStats {
                                duration_secs: duration,
                                total_frames: state_guard.total_frames,
                                file_size_bytes: state_guard.total_bytes,
                                buffer_frames: output_frames,
                                is_recording: true,
                                is_monitoring: false,
                                pre_roll_buffer_secs: 0.0,
                                left_rms_db: combined_left,
                                right_rms_db: combined_right,
                                per_source_stats: Vec::new(),
                            }
                        }
                    };
                    let _ = state_guard.event_tx.send(CaptureEvent::StatsUpdate(stats));
                    state_guard.last_status_update = Instant::now();
                }
            }
        }

        // Process commands (with timeout so we keep processing mic audio)
        match command_rx.recv_timeout(Duration::from_millis(10)) {
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

    // Signal stop to the output handlers
    stop_flag.store(true, Ordering::Relaxed);

    // Stop all streams
    for stream in &mut streams {
        let _ = stream.stop_capture();
    }

    // Wait for mic processing threads to finish
    for thread in mic_threads {
        let _ = thread.join();
    }

    // Process any remaining mic audio
    while let Ok(mic_audio) = mic_rx.try_recv() {
        if let Ok(mut state_guard) = state.lock() {
            state_guard
                .channel_buffer
                .push(mic_audio.source_index, &mic_audio.samples);
        }
    }

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
    // Validate that we have at least one audio source (PIDs or microphones)
    let has_apps = !config.pids.is_empty();
    let mic_count = config.microphones.len();
    if !has_apps && mic_count == 0 {
        return Err(anyhow!(
            "No audio source specified (no PIDs or microphones)"
        ));
    }

    // Normalize sample rate to ScreenCaptureKit supported values (8000, 16000, 24000, 48000)
    let sample_rate = crate::core::types::normalize_sample_rate(config.sample_rate);
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

    // Create content filter for app stream
    let app_filter = if apps.is_empty() {
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

    // Calculate source count: 1 for apps (if any) + N microphones
    let source_count = if has_apps { 1 } else { 0 } + mic_count;

    // Calculate output channels: 2 per source
    let output_channels: u16 = (source_count * config.channels as usize) as u16;

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

    // Create channel buffer for interleaving audio from multiple sources
    let channel_buffer = AudioChannelBuffer::new(source_count, config.channels as usize);

    // Initialize RMS tracking for each source
    let source_rms: Vec<(f32, f32)> = (0..source_count).map(|_| (-60.0, -60.0)).collect();

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
        source_rms,
    }));

    let stop_flag = Arc::new(AtomicBool::new(false));

    // Track all streams we create
    let mut streams: Vec<SCStream> = Vec::new();

    // Create the app stream (if we have apps or need a base stream for mic-only)
    if has_apps || mic_count == 0 {
        let mut app_stream_config = SCStreamConfiguration::new();
        app_stream_config
            .set_captures_audio(has_apps) // Only capture app audio if we have apps
            .set_sample_rate(sample_rate as i32)
            .set_channel_count(config.channels as i32)
            .set_width(width)
            .set_height(height);
        // Note: No microphone config on app stream - mics get their own streams

        let mut app_stream = SCStream::new(&app_filter, &app_stream_config);

        // Add handler for app audio
        if has_apps {
            let audio_handler = AudioHandler {
                state: state.clone(),
                stop_flag: stop_flag.clone(),
                source_index: 0, // Apps are always source index 0
                audio_source: AudioSourceType::App,
            };
            app_stream.add_output_handler(audio_handler, SCStreamOutputType::Audio);
        }

        streams.push(app_stream);
    }

    // Create channel for receiving processed mic audio
    let (mic_tx, mic_rx) = channel::<MicProcessedAudio>();

    // Track mic processing threads
    let mut mic_threads: Vec<JoinHandle<()>> = Vec::new();

    // Create separate streams for each microphone
    for (mic_idx, mic_config) in config.microphones.iter().enumerate() {
        // Source index: 0 if no apps, otherwise 1+
        let source_index = if has_apps { 1 + mic_idx } else { mic_idx };

        // Create shared buffer for this microphone
        let mic_buffer = Arc::new(Mutex::new(MicrophoneBuffer::new(config.channels as usize)));

        // Create minimal content filter for mic-only stream
        let mic_filter = SCContentFilter::builder().display(display).build();

        // Configure for microphone only - don't set sample rate to use native
        let mut mic_stream_config = SCStreamConfiguration::new();
        mic_stream_config
            .set_captures_audio(false) // No app audio
            .set_captures_microphone(true)
            .set_microphone_capture_device_id(&mic_config.id)
            .set_channel_count(config.channels as i32)
            .set_width(width)
            .set_height(height);
        // Note: Don't set sample_rate - let it use native rate for resampling

        let mut mic_stream = SCStream::new(&mic_filter, &mic_stream_config);

        // Add lightweight buffer handler (no resampling in callback)
        let mic_handler = MicrophoneBufferHandler {
            buffer: mic_buffer.clone(),
            stop_flag: stop_flag.clone(),
            target_channels: config.channels as usize,
        };
        mic_stream.add_output_handler(mic_handler, SCStreamOutputType::Microphone);

        streams.push(mic_stream);

        // Spawn processing thread for this microphone
        let buffer_clone = mic_buffer.clone();
        let stop_flag_clone = stop_flag.clone();
        let mic_tx_clone = mic_tx.clone();
        let channels = config.channels as usize;

        let thread_handle = thread::spawn(move || {
            mic_processing_loop(
                buffer_clone,
                stop_flag_clone,
                sample_rate,
                channels,
                source_index,
                mic_tx_clone,
            );
        });
        mic_threads.push(thread_handle);
    }

    // Drop the original sender so mic_rx will close when all threads are done
    drop(mic_tx);

    // Start all streams
    for stream in &mut streams {
        stream
            .start_capture()
            .map_err(|e| anyhow!("Failed to start capture: {:?}", e))?;
    }

    // Notify that capture has started
    let _ = event_tx.send(CaptureEvent::Started {
        buffer_size: config.sample_rate as usize / 10, // Approximate 100ms buffer
    });

    // Process commands and mic audio
    loop {
        // Process any pending mic audio (non-blocking)
        while let Ok(mic_audio) = mic_rx.try_recv() {
            if let Ok(mut state_guard) = state.lock() {
                // Update RMS for this source
                if let Some(rms) = state_guard.source_rms.get_mut(mic_audio.source_index) {
                    *rms = (mic_audio.left_rms_db, mic_audio.right_rms_db);
                }

                // Push to channel buffer
                state_guard
                    .channel_buffer
                    .push(mic_audio.source_index, &mic_audio.samples);

                // Drain and write
                let output_samples = state_guard.channel_buffer.drain_interleaved();
                let output_frames = output_samples.len() / state_guard.output_channels;

                for &sample in &output_samples {
                    let _ = state_guard.wav_writer.write_sample(sample);
                }
                if !output_samples.is_empty() {
                    let num_bytes = output_samples.len() * 4;
                    state_guard.total_frames += output_frames as u64;
                    state_guard.total_bytes += num_bytes as u64;
                }

                // Send stats update if enough time has passed (100ms)
                // This is needed when there's no app audio handler to send updates
                if state_guard.last_status_update.elapsed() >= Duration::from_millis(100) {
                    let duration = state_guard.start_time.elapsed();

                    // Combined RMS from all sources
                    let combined_left = state_guard
                        .source_rms
                        .iter()
                        .map(|(l, _)| *l)
                        .fold(-60.0_f32, f32::max);
                    let combined_right = state_guard
                        .source_rms
                        .iter()
                        .map(|(_, r)| *r)
                        .fold(-60.0_f32, f32::max);

                    let stats = CaptureStats {
                        duration_secs: duration.as_secs_f64(),
                        total_frames: state_guard.total_frames,
                        file_size_bytes: state_guard.total_bytes,
                        buffer_frames: output_frames,
                        is_recording: true,
                        is_monitoring: false,
                        pre_roll_buffer_secs: 0.0,
                        left_rms_db: combined_left,
                        right_rms_db: combined_right,
                        per_source_stats: Vec::new(),
                    };
                    let _ = event_tx.send(CaptureEvent::StatsUpdate(stats));
                    state_guard.last_status_update = Instant::now();
                }
            }
        }

        // Process commands (with timeout so we keep processing mic audio)
        match command_rx.recv_timeout(Duration::from_millis(10)) {
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

    // Signal stop to the output handlers
    stop_flag.store(true, Ordering::Relaxed);

    // Stop all streams
    for stream in &mut streams {
        let _ = stream.stop_capture();
    }

    // Wait for mic processing threads to finish
    for thread in mic_threads {
        let _ = thread.join();
    }

    // Process any remaining mic audio
    while let Ok(mic_audio) = mic_rx.try_recv() {
        if let Ok(mut state_guard) = state.lock() {
            state_guard
                .channel_buffer
                .push(mic_audio.source_index, &mic_audio.samples);
        }
    }

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
