//! Windows audio capture implementation using WASAPI

use crate::core::process::process_exists;
use crate::core::ring_buffer::AudioRingBuffer;
use crate::core::types::{
    CaptureCommand, CaptureConfig, CaptureEvent, CaptureStats, MonitorConfig, SourceStats,
};
use anyhow::{anyhow, Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use std::collections::VecDeque;
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use wasapi::{AudioClient, DeviceCollection, Direction, SampleType, StreamMode, WaveFormat};

/// Capture mode for the session
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CaptureMode {
    /// Capturing to ring buffer only (pre-recording)
    Monitoring,
    /// Capturing to WAV file
    Recording,
}

/// Manages an audio capture session on Windows
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
    // Initialize COM in this thread
    let hr = wasapi::initialize_mta();
    if hr.is_err() {
        let _ = event_tx.send(CaptureEvent::Error(format!(
            "Failed to initialize COM: HRESULT {:#x}",
            hr.0
        )));
        return;
    }

    // Run the capture and handle any errors
    if let Err(e) = run_capture(&config, &command_rx, &event_tx) {
        let _ = event_tx.send(CaptureEvent::Error(e.to_string()));
    }

    let _ = event_tx.send(CaptureEvent::Stopped);
}

/// Holds a WASAPI audio client and its capture client for one process (app loopback)
struct AudioClientHandle {
    audio_client: AudioClient,
    capture_client: wasapi::AudioCaptureClient,
    event_handle: wasapi::Handle,
    audio_buffer: VecDeque<u8>,
}

/// Holds a WASAPI audio client for microphone capture (standard capture, not loopback)
struct MicrophoneClientHandle {
    audio_client: AudioClient,
    capture_client: wasapi::AudioCaptureClient,
    event_handle: wasapi::Handle,
    audio_buffer: VecDeque<u8>,
}

/// Buffer that accumulates audio from multiple sources and interleaves them.
///
/// Each source gets its own stereo pair in the output:
/// - Source 0: channels 0-1
/// - Source 1: channels 2-3
/// - etc.
///
/// Output format: `[src0_L, src0_R, src1_L, src1_R, ...]` per frame
struct MultiSourceBuffer {
    /// Per-source sample buffers (interleaved stereo samples)
    source_buffers: Vec<VecDeque<f32>>,
    /// Number of sources
    source_count: usize,
    /// Channels per source (always 2 for stereo)
    channels_per_source: usize,
    /// Minimum samples to buffer before starting output (for sync across sources)
    /// This is per-source, in samples (not frames)
    min_buffer_samples: usize,
    /// Whether we've started outputting (after initial sync delay)
    started: bool,
}

impl MultiSourceBuffer {
    /// Create a new multi-source buffer
    ///
    /// # Arguments
    /// * `source_count` - Number of audio sources (apps + microphone)
    /// * `sample_rate` - Sample rate in Hz (used to calculate sync delay)
    fn new(source_count: usize, sample_rate: u32) -> Self {
        // 50ms sync delay at given sample rate, in stereo samples
        let min_buffer_samples = (sample_rate as usize * 2 * 50) / 1000;

        Self {
            source_buffers: (0..source_count).map(|_| VecDeque::new()).collect(),
            source_count,
            channels_per_source: 2,
            min_buffer_samples,
            started: false,
        }
    }

    /// Push interleaved stereo samples for a specific source
    fn push(&mut self, source_index: usize, samples: &[f32]) {
        if source_index < self.source_count {
            self.source_buffers[source_index].extend(samples);
        }
    }

    /// Check if all sources have enough data to start outputting
    fn all_sources_ready(&self) -> bool {
        self.source_buffers
            .iter()
            .all(|buf| buf.len() >= self.min_buffer_samples)
    }

    /// Get the minimum number of complete frames available across all sources
    fn min_frames_available(&self) -> usize {
        self.source_buffers
            .iter()
            .map(|buf| buf.len() / self.channels_per_source)
            .min()
            .unwrap_or(0)
    }

    /// Drain interleaved output from all sources
    ///
    /// Returns samples in format: `[src0_L, src0_R, src1_L, src1_R, ...]` per frame
    /// Only returns complete frames where all sources have data.
    fn drain_interleaved(&mut self) -> Vec<f32> {
        // For single source, no sync delay needed
        if self.source_count == 1 {
            self.started = true;
            let samples: Vec<f32> = self.source_buffers[0].drain(..).collect();
            return samples;
        }

        // Wait for initial sync before starting output
        if !self.started {
            if self.all_sources_ready() {
                self.started = true;
            } else {
                return Vec::new();
            }
        }

        let frames_to_drain = self.min_frames_available();
        if frames_to_drain == 0 {
            return Vec::new();
        }

        let total_output_samples = frames_to_drain * self.channels_per_source * self.source_count;
        let mut output = Vec::with_capacity(total_output_samples);

        // Interleave: for each frame, output all sources' samples
        for _frame in 0..frames_to_drain {
            for source in &mut self.source_buffers {
                // Output L and R for this source
                let l = source.pop_front().unwrap_or(0.0);
                let r = source.pop_front().unwrap_or(0.0);
                output.push(l);
                output.push(r);
            }
        }

        output
    }

    /// Get the total output channel count
    fn output_channels(&self) -> u16 {
        (self.source_count * self.channels_per_source) as u16
    }
}

/// Shared capture context for both monitoring and recording modes.
///
/// Encapsulates client setup, polling, and sample reading logic to avoid
/// code duplication between `run_monitor` (which saves to memory) and `run_capture` (which saves to disk).
struct CaptureContext {
    /// WASAPI clients for app audio (one per PID)
    app_clients: Vec<AudioClientHandle>,
    /// Optional microphone client
    mic_client: Option<MicrophoneClientHandle>,
    /// Multi-source buffer for interleaving
    source_buffer: MultiSourceBuffer,
    /// Output channel count
    output_channels: u16,
    /// Sample rate
    sample_rate: u32,
    /// PIDs for per-source stats (apps first, then mic uses PID 0)
    pids: Vec<u32>,
}

impl CaptureContext {
    /// Create a new capture context with the given sources
    fn new(
        pids: &[u32],
        microphone_id: Option<&str>,
        sample_rate: u32,
        channels: u16,
    ) -> Result<Self> {
        let has_apps = !pids.is_empty();
        let has_mic = microphone_id.is_some();

        // Validate at least one source
        if !has_apps && !has_mic {
            return Err(anyhow!("No audio source specified"));
        }

        // Validate that all processes exist
        for &pid in pids {
            if !process_exists(pid) {
                return Err(anyhow!("Process with PID {} does not exist", pid));
            }
        }

        // Define the desired audio format
        let desired_format = WaveFormat::new(
            32, // bits per sample
            32, // valid bits
            &SampleType::Float,
            sample_rate as usize,
            channels as usize,
            None,
        );

        let buffer_duration_hns = 2_000_000i64; // 200ms
        let mode = StreamMode::EventsShared {
            autoconvert: true,
            buffer_duration_hns,
        };

        // Create loopback clients for all PIDs
        let mut app_clients: Vec<AudioClientHandle> = Vec::new();
        for &pid in pids {
            let mut audio_client =
                AudioClient::new_application_loopback_client(pid, true).map_err(|e| {
                    anyhow!(
                        "Failed to create application loopback client for PID {}: {}",
                        pid,
                        e
                    )
                })?;

            audio_client
                .initialize_client(&desired_format, &Direction::Capture, &mode)
                .map_err(|e| {
                    anyhow!("Failed to initialize audio client for PID {}: {}", pid, e)
                })?;

            let event_handle = audio_client
                .set_get_eventhandle()
                .map_err(|e| anyhow!("Failed to get event handle for PID {}: {}", pid, e))?;

            let capture_client = audio_client
                .get_audiocaptureclient()
                .map_err(|e| anyhow!("Failed to get audio capture client for PID {}: {}", pid, e))?;

            app_clients.push(AudioClientHandle {
                audio_client,
                capture_client,
                event_handle,
                audio_buffer: VecDeque::new(),
            });
        }

        // Create microphone client if specified
        let mic_client = if let Some(mic_id) = microphone_id {
            Some(create_microphone_client(mic_id, sample_rate, channels)?)
        } else {
            None
        };

        // Calculate total source count and create buffer
        let source_count = pids.len() + if has_mic { 1 } else { 0 };
        let source_buffer = MultiSourceBuffer::new(source_count, sample_rate);
        let output_channels = source_buffer.output_channels();

        // Build PID list for stats (apps first, mic uses PID 0)
        let mut stat_pids: Vec<u32> = pids.to_vec();
        if has_mic {
            stat_pids.push(0); // PID 0 represents microphone
        }

        Ok(Self {
            app_clients,
            mic_client,
            source_buffer,
            output_channels,
            sample_rate,
            pids: stat_pids,
        })
    }

    /// Start all audio streams
    fn start_streams(&self) -> Result<()> {
        for client in &self.app_clients {
            client
                .audio_client
                .start_stream()
                .map_err(|e| anyhow!("Failed to start app audio stream: {}", e))?;
        }

        if let Some(ref mic) = self.mic_client {
            mic.audio_client
                .start_stream()
                .map_err(|e| anyhow!("Failed to start microphone stream: {}", e))?;
        }

        Ok(())
    }

    /// Stop all audio streams
    fn stop_streams(&self) {
        for client in &self.app_clients {
            let _ = client.audio_client.stop_stream();
        }

        if let Some(ref mic) = self.mic_client {
            let _ = mic.audio_client.stop_stream();
        }
    }

    /// Poll all sources for audio data, read samples, and return interleaved output + per-source stats
    fn poll_and_read(&mut self) -> (Vec<f32>, Vec<SourceStats>) {
        // Wait for audio data from any client
        let mut any_data = false;

        for client in &self.app_clients {
            if client.event_handle.wait_for_event(5).is_ok() {
                any_data = true;
            }
        }

        if let Some(ref mic) = self.mic_client {
            if mic.event_handle.wait_for_event(5).is_ok() {
                any_data = true;
            }
        }

        if !any_data {
            return (Vec::new(), Vec::new());
        }

        // Per-source samples for RMS calculation
        let mut per_source_samples: Vec<(u32, Vec<f32>)> = Vec::new();

        // Read from all app clients
        for (client_idx, client) in self.app_clients.iter_mut().enumerate() {
            let samples = Self::read_client_samples(client);
            if !samples.is_empty() {
                per_source_samples.push((self.pids[client_idx], samples.clone()));
                self.source_buffer.push(client_idx, &samples);
            }
        }

        // Read from microphone client
        if let Some(ref mut mic) = self.mic_client {
            let samples = Self::read_mic_samples(mic);
            if !samples.is_empty() {
                let mic_source_idx = self.app_clients.len();
                per_source_samples.push((0, samples.clone())); // PID 0 for mic
                self.source_buffer.push(mic_source_idx, &samples);
            }
        }

        // Get interleaved output
        let mut output = self.source_buffer.drain_interleaved();

        // Clamp samples to prevent clipping
        for sample in &mut output {
            *sample = sample.clamp(-1.0, 1.0);
        }

        // Calculate per-source RMS stats
        let per_source_stats = self.calculate_per_source_stats(&per_source_samples);

        (output, per_source_stats)
    }

    /// Read samples from an app loopback client
    fn read_client_samples(client: &mut AudioClientHandle) -> Vec<f32> {
        // Check how many frames are available
        match client.capture_client.get_next_packet_size() {
            Ok(Some(frames)) if frames > 0 => {}
            _ => return Vec::new(),
        }

        // Read available frames
        client.audio_buffer.clear();
        match client
            .capture_client
            .read_from_device_to_deque(&mut client.audio_buffer)
        {
            Ok(_) => {
                let bytes = client.audio_buffer.make_contiguous();
                bytes_to_f32_samples(bytes).to_vec()
            }
            Err(e) => {
                let err_str = format!("{}", e);
                if err_str.contains("AUDCLNT_S_BUFFER_EMPTY") || err_str.contains("0x08890001") {
                    Vec::new()
                } else {
                    Vec::new() // Silently ignore other errors
                }
            }
        }
    }

    /// Read samples from the microphone client
    fn read_mic_samples(mic: &mut MicrophoneClientHandle) -> Vec<f32> {
        // Check how many frames are available
        match mic.capture_client.get_next_packet_size() {
            Ok(Some(frames)) if frames > 0 => {}
            _ => return Vec::new(),
        }

        // Read available frames
        mic.audio_buffer.clear();
        match mic
            .capture_client
            .read_from_device_to_deque(&mut mic.audio_buffer)
        {
            Ok(_) => {
                let bytes = mic.audio_buffer.make_contiguous();
                bytes_to_f32_samples(bytes).to_vec()
            }
            Err(e) => {
                let err_str = format!("{}", e);
                if err_str.contains("AUDCLNT_S_BUFFER_EMPTY") || err_str.contains("0x08890001") {
                    Vec::new()
                } else {
                    Vec::new() // Silently ignore other errors
                }
            }
        }
    }

    /// Calculate RMS stats for each source
    fn calculate_per_source_stats(&self, per_source_samples: &[(u32, Vec<f32>)]) -> Vec<SourceStats> {
        per_source_samples
            .iter()
            .map(|(pid, samples)| {
                let left: Vec<f32> = samples.iter().step_by(2).copied().collect();
                let right: Vec<f32> = if samples.len() > 1 {
                    samples.iter().skip(1).step_by(2).copied().collect()
                } else {
                    left.clone()
                };
                SourceStats {
                    pid: *pid,
                    left_rms_db: calculate_rms_db(&left),
                    right_rms_db: calculate_rms_db(&right),
                }
            })
            .collect()
    }

    /// Calculate combined RMS from interleaved output samples
    fn calculate_combined_rms(&self, samples: &[f32]) -> (f32, f32) {
        if samples.is_empty() {
            return (-60.0, -60.0);
        }

        // For multi-channel output, combine all even channels (lefts) and odd channels (rights)
        let channels = self.output_channels as usize;
        let mut left_samples = Vec::new();
        let mut right_samples = Vec::new();

        for (i, &sample) in samples.iter().enumerate() {
            let channel_in_frame = i % channels;
            if channel_in_frame % 2 == 0 {
                left_samples.push(sample);
            } else {
                right_samples.push(sample);
            }
        }

        (
            calculate_rms_db(&left_samples),
            calculate_rms_db(&right_samples),
        )
    }
}

/// Create a microphone capture client for the given device ID
fn create_microphone_client(
    device_id: &str,
    sample_rate: u32,
    channels: u16,
) -> Result<MicrophoneClientHandle> {
    // Enumerate capture devices and find the one matching our ID
    let collection = DeviceCollection::new(&Direction::Capture)
        .map_err(|e| anyhow!("Failed to enumerate capture devices: {}", e))?;

    let device = collection
        .into_iter()
        .filter_map(|d| d.ok())
        .find(|d| d.get_id().ok().as_ref() == Some(&device_id.to_string()))
        .ok_or_else(|| anyhow!("Microphone device not found: {}", device_id))?;

    // Get audio client from device
    let mut audio_client = device
        .get_iaudioclient()
        .map_err(|e| anyhow!("Failed to get audio client for microphone: {}", e))?;

    // Configure format - same as app capture
    let desired_format = WaveFormat::new(
        32, // bits per sample
        32, // valid bits
        &SampleType::Float,
        sample_rate as usize,
        channels as usize,
        None,
    );

    // Initialize for standard capture (not loopback)
    let buffer_duration_hns = 2_000_000i64; // 200ms
    let mode = StreamMode::EventsShared {
        autoconvert: true,
        buffer_duration_hns,
    };

    audio_client
        .initialize_client(&desired_format, &Direction::Capture, &mode)
        .map_err(|e| anyhow!("Failed to initialize microphone audio client: {}", e))?;

    let event_handle = audio_client
        .set_get_eventhandle()
        .map_err(|e| anyhow!("Failed to get event handle for microphone: {}", e))?;

    let capture_client = audio_client
        .get_audiocaptureclient()
        .map_err(|e| anyhow!("Failed to get capture client for microphone: {}", e))?;

    Ok(MicrophoneClientHandle {
        audio_client,
        capture_client,
        event_handle,
        audio_buffer: VecDeque::new(),
    })
}

fn run_capture(
    config: &CaptureConfig,
    command_rx: &Receiver<CaptureCommand>,
    event_tx: &Sender<CaptureEvent>,
) -> Result<()> {
    // Create capture context with apps and optional microphone
    let mut ctx = CaptureContext::new(
        &config.pids,
        config.microphone_id.as_deref(),
        config.sample_rate,
        config.channels,
    )?;

    // Notify UI that capture has started
    let _ = event_tx.send(CaptureEvent::Started { buffer_size: 4800 });

    // Set up WAV writer with output channel count (per-source channels)
    let wav_spec = WavSpec {
        channels: ctx.output_channels,
        sample_rate: config.sample_rate,
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
    let mut wav_writer =
        WavWriter::new(buf_writer, wav_spec).context("Failed to create WAV writer")?;

    // Start all streams
    ctx.start_streams()?;

    let start_time = Instant::now();
    let mut total_frames: u64 = 0;
    let mut total_bytes: u64 = 0;
    let mut last_status_update = Instant::now();
    let mut last_samples: Vec<f32> = Vec::new();
    let mut last_per_source_stats: Vec<SourceStats> = Vec::new();

    // Main capture loop
    loop {
        // Check for commands (non-blocking)
        match command_rx.try_recv() {
            Ok(CaptureCommand::Stop) => break,
            Ok(CaptureCommand::StartRecording { .. }) => {
                // Ignored in direct capture mode - already recording
            }
            Ok(CaptureCommand::ResizePreRoll { .. }) => {
                // Ignored in direct capture mode - no ring buffer
            }
            Err(_) => {}
        }

        // Poll and read from all sources (apps + microphone)
        let (samples, per_source_stats) = ctx.poll_and_read();

        if samples.is_empty() {
            continue;
        }

        // Store for stats calculation
        last_samples = samples.clone();
        last_per_source_stats = per_source_stats;

        // Write samples to WAV
        for &sample in &samples {
            wav_writer
                .write_sample(sample)
                .context("Failed to write sample to WAV")?;
        }

        let num_bytes = samples.len() * 4;
        let num_frames = num_bytes / (ctx.output_channels as usize * 4);
        total_frames += num_frames as u64;
        total_bytes += num_bytes as u64;

        // Update status every 100ms
        if last_status_update.elapsed() >= Duration::from_millis(100) {
            let duration = start_time.elapsed();
            let (left_rms_db, right_rms_db) = ctx.calculate_combined_rms(&last_samples);

            let stats = CaptureStats {
                duration_secs: duration.as_secs_f64(),
                total_frames,
                file_size_bytes: total_bytes,
                buffer_frames: num_frames,
                is_recording: true,
                is_monitoring: false,
                pre_roll_buffer_secs: 0.0,
                left_rms_db,
                right_rms_db,
                per_source_stats: last_per_source_stats.clone(),
            };
            let _ = event_tx.send(CaptureEvent::StatsUpdate(stats));
            last_status_update = Instant::now();
        }
    }

    // Stop all streams
    ctx.stop_streams();

    // Finalize the WAV file
    wav_writer
        .finalize()
        .context("Failed to finalize WAV file")?;

    Ok(())
}

/// Main entry point for monitoring thread
fn monitor_thread_main(
    config: MonitorConfig,
    command_rx: Receiver<CaptureCommand>,
    event_tx: Sender<CaptureEvent>,
) {
    // Initialize COM in this thread
    let hr = wasapi::initialize_mta();
    if hr.is_err() {
        let _ = event_tx.send(CaptureEvent::Error(format!(
            "Failed to initialize COM: HRESULT {:#x}",
            hr.0
        )));
        return;
    }

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
    // Create capture context with apps and optional microphone
    let mut ctx = CaptureContext::new(
        &config.pids,
        config.microphone_id.as_deref(),
        config.sample_rate,
        config.channels,
    )?;

    // Create ring buffer with output channel count (per-source channels)
    let mut ring_buffer = AudioRingBuffer::new(
        config.pre_roll_duration_secs,
        config.sample_rate,
        ctx.output_channels,
    );

    // WAV spec for when we transition to recording (uses output channels from context)
    let wav_spec = WavSpec {
        channels: ctx.output_channels,
        sample_rate: config.sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    // Start all streams
    ctx.start_streams()?;

    // Notify that monitoring has started
    let _ = event_tx.send(CaptureEvent::MonitoringStarted);

    let start_time = Instant::now();
    let mut last_status_update = Instant::now();
    let mut capture_mode = CaptureMode::Monitoring;
    let mut wav_writer: Option<WavWriter<BufWriter<std::fs::File>>> = None;
    let mut total_frames: u64 = 0;
    let mut total_bytes: u64 = 0;
    let mut recording_start_time: Option<Instant> = None;
    let mut last_samples: Vec<f32> = Vec::new();
    let mut last_per_source_stats: Vec<SourceStats> = Vec::new();

    // Main capture loop
    loop {
        // Check for commands (non-blocking)
        match command_rx.try_recv() {
            Ok(CaptureCommand::Stop) => break,
            Ok(CaptureCommand::StartRecording { output_path }) => {
                // Transition to recording mode
                let file = std::fs::File::create(&output_path).with_context(|| {
                    format!("Failed to create output file: {}", output_path.display())
                })?;
                let buf_writer = BufWriter::new(file);
                let mut writer =
                    WavWriter::new(buf_writer, wav_spec).context("Failed to create WAV writer")?;

                // Drain ring buffer and write to WAV
                let pre_roll_secs = ring_buffer.duration_secs();
                let samples = ring_buffer.drain();
                for sample in samples {
                    writer.write_sample(sample)?;
                }

                wav_writer = Some(writer);
                capture_mode = CaptureMode::Recording;
                recording_start_time = Some(Instant::now());
                total_frames = 0;
                total_bytes = 0;

                let _ = event_tx.send(CaptureEvent::RecordingStarted { pre_roll_secs });
            }
            Ok(CaptureCommand::ResizePreRoll { duration_secs }) => {
                ring_buffer.resize(duration_secs);
            }
            Err(_) => {}
        }

        // Poll and read from all sources (apps + microphone)
        let (samples, per_source_stats) = ctx.poll_and_read();

        if samples.is_empty() {
            continue;
        }

        // Store for stats calculation
        last_samples = samples.clone();
        last_per_source_stats = per_source_stats;

        let num_bytes = samples.len() * 4;
        let num_frames = num_bytes / (ctx.output_channels as usize * 4);

        match capture_mode {
            CaptureMode::Monitoring => {
                // Push to ring buffer
                ring_buffer.push(&samples);
            }
            CaptureMode::Recording => {
                // Write to WAV file
                if let Some(ref mut writer) = wav_writer {
                    for &sample in &samples {
                        writer
                            .write_sample(sample)
                            .context("Failed to write sample to WAV")?;
                    }
                }
                total_frames += num_frames as u64;
                total_bytes += num_bytes as u64;
            }
        }

        // Update status every 100ms
        if last_status_update.elapsed() >= Duration::from_millis(100) {
            let (left_rms_db, right_rms_db) = ctx.calculate_combined_rms(&last_samples);

            let stats = match capture_mode {
                CaptureMode::Monitoring => CaptureStats {
                    duration_secs: start_time.elapsed().as_secs_f64(),
                    total_frames: 0,
                    file_size_bytes: 0,
                    buffer_frames: num_frames,
                    is_recording: false,
                    is_monitoring: true,
                    pre_roll_buffer_secs: ring_buffer.duration_secs(),
                    left_rms_db,
                    right_rms_db,
                    per_source_stats: last_per_source_stats.clone(),
                },
                CaptureMode::Recording => {
                    let duration = recording_start_time
                        .map(|t| t.elapsed().as_secs_f64())
                        .unwrap_or(0.0);
                    CaptureStats {
                        duration_secs: duration,
                        total_frames,
                        file_size_bytes: total_bytes,
                        buffer_frames: num_frames,
                        is_recording: true,
                        is_monitoring: false,
                        pre_roll_buffer_secs: 0.0,
                        left_rms_db,
                        right_rms_db,
                        per_source_stats: last_per_source_stats.clone(),
                    }
                }
            };
            let _ = event_tx.send(CaptureEvent::StatsUpdate(stats));
            last_status_update = Instant::now();
        }
    }

    // Stop all streams
    ctx.stop_streams();

    // Finalize WAV file if we were recording
    if let Some(writer) = wav_writer {
        writer.finalize().context("Failed to finalize WAV file")?;
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
