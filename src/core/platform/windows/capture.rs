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
use wasapi::{AudioClient, Direction, SampleType, StreamMode, WaveFormat};

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

/// Holds a WASAPI audio client and its capture client for one process
struct AudioClientHandle {
    audio_client: AudioClient,
    capture_client: wasapi::AudioCaptureClient,
    event_handle: wasapi::Handle,
    audio_buffer: VecDeque<u8>,
}

fn run_capture(
    config: &CaptureConfig,
    command_rx: &Receiver<CaptureCommand>,
    event_tx: &Sender<CaptureEvent>,
) -> Result<()> {
    // Validate that we have at least one PID
    if config.pids.is_empty() {
        return Err(anyhow!("No PIDs specified for capture"));
    }

    // Validate that all processes exist
    for &pid in &config.pids {
        if !process_exists(pid) {
            return Err(anyhow!("Process with PID {} does not exist", pid));
        }
    }

    // Define the desired audio format
    let block_align = (config.channels * config.bits_per_sample / 8) as u16;

    let desired_format = WaveFormat::new(
        config.bits_per_sample as usize,
        config.bits_per_sample as usize,
        &SampleType::Float,
        config.sample_rate as usize,
        config.channels as usize,
        None,
    );

    // Initialize the audio client
    let buffer_duration_hns = 2_000_000i64; // 200ms
    let mode = StreamMode::EventsShared {
        autoconvert: true,
        buffer_duration_hns,
    };

    // Create loopback clients for all PIDs
    let mut clients: Vec<AudioClientHandle> = Vec::new();
    for &pid in &config.pids {
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
            .map_err(|e| anyhow!("Failed to initialize audio client for PID {}: {}", pid, e))?;

        let event_handle = audio_client
            .set_get_eventhandle()
            .map_err(|e| anyhow!("Failed to get event handle for PID {}: {}", pid, e))?;

        let capture_client = audio_client
            .get_audiocaptureclient()
            .map_err(|e| anyhow!("Failed to get audio capture client for PID {}: {}", pid, e))?;

        clients.push(AudioClientHandle {
            audio_client,
            capture_client,
            event_handle,
            audio_buffer: VecDeque::new(),
        });
    }

    let buffer_size = clients
        .first()
        .and_then(|c| c.audio_client.get_buffer_size().ok())
        .unwrap_or(4800);

    // Notify UI that capture has started
    let _ = event_tx.send(CaptureEvent::Started {
        buffer_size: buffer_size as usize,
    });

    // Set up WAV writer
    let wav_spec = WavSpec {
        channels: config.channels,
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
    for client in &clients {
        client
            .audio_client
            .start_stream()
            .map_err(|e| anyhow!("Failed to start audio stream: {}", e))?;
    }

    let start_time = Instant::now();
    let mut total_frames: u64 = 0;
    let mut total_bytes: u64 = 0;
    let mut last_status_update = Instant::now();
    #[allow(unused_assignments)]
    let mut current_buffer_frames = 0usize;
    let mut mixed_samples: Vec<f32> = Vec::new();

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

        // Wait for audio data from any client (timeout 100ms)
        // We poll each client since we can't easily wait on multiple handles
        let mut any_data = false;
        for client in &clients {
            if client.event_handle.wait_for_event(10).is_ok() {
                any_data = true;
                break;
            }
        }
        if !any_data {
            continue;
        }

        // Read from all clients and mix
        mixed_samples.clear();
        let mut max_samples = 0usize;
        // Store per-source samples for RMS calculation
        let mut per_source_samples: Vec<(u32, Vec<f32>)> = Vec::new();

        for (client_idx, client) in clients.iter_mut().enumerate() {
            // Check how many frames are available
            let frames_available = match client.capture_client.get_next_packet_size() {
                Ok(Some(frames)) => frames,
                Ok(None) => continue,
                Err(_) => continue,
            };

            if frames_available == 0 {
                continue;
            }

            // Read available frames
            client.audio_buffer.clear();
            match client
                .capture_client
                .read_from_device_to_deque(&mut client.audio_buffer)
            {
                Ok(_buffer_flags) => {
                    let num_bytes = client.audio_buffer.len();
                    if num_bytes == 0 {
                        continue;
                    }

                    let bytes = client.audio_buffer.make_contiguous();
                    let samples = bytes_to_f32_samples(bytes);

                    // Store samples for per-source RMS calculation
                    per_source_samples.push((config.pids[client_idx], samples.to_vec()));

                    // Mix into the combined buffer
                    if samples.len() > max_samples {
                        // Extend mixed_samples to accommodate more samples
                        mixed_samples.resize(samples.len(), 0.0);
                        max_samples = samples.len();
                    }

                    for (i, &sample) in samples.iter().enumerate() {
                        mixed_samples[i] += sample;
                    }
                }
                Err(e) => {
                    let err_str = format!("{}", e);
                    if err_str.contains("AUDCLNT_S_BUFFER_EMPTY") || err_str.contains("0x08890001")
                    {
                        continue;
                    }
                    return Err(anyhow!("Error reading audio data: {}", e));
                }
            }
        }

        if mixed_samples.is_empty() {
            continue;
        }

        // Clamp mixed samples to prevent clipping
        for sample in &mut mixed_samples {
            *sample = sample.clamp(-1.0, 1.0);
        }

        // Write mixed samples to WAV
        for &sample in &mixed_samples {
            wav_writer
                .write_sample(sample)
                .context("Failed to write sample to WAV")?;
        }

        let num_bytes = mixed_samples.len() * 4;
        let num_frames = num_bytes / block_align as usize;
        total_frames += num_frames as u64;
        total_bytes += num_bytes as u64;
        current_buffer_frames = num_frames;

        // Update status every 100ms
        if last_status_update.elapsed() >= Duration::from_millis(100) {
            let duration = start_time.elapsed();
            // Separate interleaved samples into channels and calculate RMS
            let num_channels = config.channels as usize;
            let left_samples: Vec<f32> = mixed_samples
                .iter()
                .step_by(num_channels)
                .copied()
                .collect();
            let right_samples: Vec<f32> = if num_channels > 1 {
                mixed_samples
                    .iter()
                    .skip(1)
                    .step_by(num_channels)
                    .copied()
                    .collect()
            } else {
                left_samples.clone()
            };
            let left_rms_db = calculate_rms_db(&left_samples);
            let right_rms_db = calculate_rms_db(&right_samples);

            // Calculate per-source RMS
            let per_source_stats: Vec<SourceStats> = per_source_samples
                .iter()
                .map(|(pid, samples)| {
                    let left: Vec<f32> = samples.iter().step_by(num_channels).copied().collect();
                    let right: Vec<f32> = if num_channels > 1 {
                        samples
                            .iter()
                            .skip(1)
                            .step_by(num_channels)
                            .copied()
                            .collect()
                    } else {
                        left.clone()
                    };
                    SourceStats {
                        pid: *pid,
                        left_rms_db: calculate_rms_db(&left),
                        right_rms_db: calculate_rms_db(&right),
                    }
                })
                .collect();

            let stats = CaptureStats {
                duration_secs: duration.as_secs_f64(),
                total_frames,
                file_size_bytes: total_bytes,
                buffer_frames: current_buffer_frames,
                is_recording: true,
                is_monitoring: false,
                pre_roll_buffer_secs: 0.0,
                left_rms_db,
                right_rms_db,
                per_source_stats,
            };
            let _ = event_tx.send(CaptureEvent::StatsUpdate(stats));
            last_status_update = Instant::now();
        }
    }

    // Stop all streams
    for client in &clients {
        let _ = client.audio_client.stop_stream();
    }

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
    // Validate that we have at least one PID
    if config.pids.is_empty() {
        return Err(anyhow!("No PIDs specified for capture"));
    }

    // Validate that all processes exist
    for &pid in &config.pids {
        if !process_exists(pid) {
            return Err(anyhow!("Process with PID {} does not exist", pid));
        }
    }

    // Define the desired audio format
    let bits_per_sample = 32u16;
    let block_align = (config.channels * bits_per_sample / 8) as u16;

    let desired_format = WaveFormat::new(
        bits_per_sample as usize,
        bits_per_sample as usize,
        &SampleType::Float,
        config.sample_rate as usize,
        config.channels as usize,
        None,
    );

    // Initialize the audio client
    let buffer_duration_hns = 2_000_000i64; // 200ms
    let mode = StreamMode::EventsShared {
        autoconvert: true,
        buffer_duration_hns,
    };

    // Create loopback clients for all PIDs
    let mut clients: Vec<AudioClientHandle> = Vec::new();
    for &pid in &config.pids {
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
            .map_err(|e| anyhow!("Failed to initialize audio client for PID {}: {}", pid, e))?;

        let event_handle = audio_client
            .set_get_eventhandle()
            .map_err(|e| anyhow!("Failed to get event handle for PID {}: {}", pid, e))?;

        let capture_client = audio_client
            .get_audiocaptureclient()
            .map_err(|e| anyhow!("Failed to get audio capture client for PID {}: {}", pid, e))?;

        clients.push(AudioClientHandle {
            audio_client,
            capture_client,
            event_handle,
            audio_buffer: VecDeque::new(),
        });
    }

    // Create ring buffer for pre-roll
    let mut ring_buffer = AudioRingBuffer::new(
        config.pre_roll_duration_secs,
        config.sample_rate,
        config.channels,
    );

    // WAV spec for when we transition to recording
    let wav_spec = WavSpec {
        channels: config.channels,
        sample_rate: config.sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    // Start all streams
    for client in &clients {
        client
            .audio_client
            .start_stream()
            .map_err(|e| anyhow!("Failed to start audio stream: {}", e))?;
    }

    // Notify that monitoring has started
    let _ = event_tx.send(CaptureEvent::MonitoringStarted);

    let start_time = Instant::now();
    let mut last_status_update = Instant::now();
    let mut capture_mode = CaptureMode::Monitoring;
    let mut wav_writer: Option<WavWriter<BufWriter<std::fs::File>>> = None;
    let mut total_frames: u64 = 0;
    let mut total_bytes: u64 = 0;
    let mut recording_start_time: Option<Instant> = None;
    let mut mixed_samples: Vec<f32> = Vec::new();

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

        // Wait for audio data from any client (timeout 100ms)
        // We poll each client since we can't easily wait on multiple handles
        let mut any_data = false;
        for client in &clients {
            if client.event_handle.wait_for_event(10).is_ok() {
                any_data = true;
                break;
            }
        }
        if !any_data {
            continue;
        }

        // Read from all clients and mix
        mixed_samples.clear();
        let mut max_samples = 0usize;
        // Store per-source samples for RMS calculation
        let mut per_source_samples: Vec<(u32, Vec<f32>)> = Vec::new();

        for (client_idx, client) in clients.iter_mut().enumerate() {
            // Check how many frames are available
            let frames_available = match client.capture_client.get_next_packet_size() {
                Ok(Some(frames)) => frames,
                Ok(None) => continue,
                Err(_) => continue,
            };

            if frames_available == 0 {
                continue;
            }

            // Read available frames
            client.audio_buffer.clear();
            match client
                .capture_client
                .read_from_device_to_deque(&mut client.audio_buffer)
            {
                Ok(_buffer_flags) => {
                    let num_bytes = client.audio_buffer.len();
                    if num_bytes == 0 {
                        continue;
                    }

                    let bytes = client.audio_buffer.make_contiguous();
                    let samples = bytes_to_f32_samples(bytes);

                    // Store samples for per-source RMS calculation
                    per_source_samples.push((config.pids[client_idx], samples.to_vec()));

                    // Mix into the combined buffer
                    if samples.len() > max_samples {
                        mixed_samples.resize(samples.len(), 0.0);
                        max_samples = samples.len();
                    }

                    for (i, &sample) in samples.iter().enumerate() {
                        mixed_samples[i] += sample;
                    }
                }
                Err(e) => {
                    let err_str = format!("{}", e);
                    if err_str.contains("AUDCLNT_S_BUFFER_EMPTY") || err_str.contains("0x08890001")
                    {
                        continue;
                    }
                    return Err(anyhow!("Error reading audio data: {}", e));
                }
            }
        }

        if mixed_samples.is_empty() {
            continue;
        }

        // Clamp mixed samples to prevent clipping
        for sample in &mut mixed_samples {
            *sample = sample.clamp(-1.0, 1.0);
        }

        let num_bytes = mixed_samples.len() * 4;
        let num_frames = num_bytes / block_align as usize;

        match capture_mode {
            CaptureMode::Monitoring => {
                // Push to ring buffer
                ring_buffer.push(&mixed_samples);
            }
            CaptureMode::Recording => {
                // Write to WAV file
                if let Some(ref mut writer) = wav_writer {
                    for &sample in &mixed_samples {
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
            let num_channels = config.channels as usize;
            let left_samples: Vec<f32> = mixed_samples
                .iter()
                .step_by(num_channels)
                .copied()
                .collect();
            let right_samples: Vec<f32> = if num_channels > 1 {
                mixed_samples
                    .iter()
                    .skip(1)
                    .step_by(num_channels)
                    .copied()
                    .collect()
            } else {
                left_samples.clone()
            };
            let left_rms_db = calculate_rms_db(&left_samples);
            let right_rms_db = calculate_rms_db(&right_samples);

            // Calculate per-source RMS
            let per_source_stats: Vec<SourceStats> = per_source_samples
                .iter()
                .map(|(pid, samples)| {
                    let left: Vec<f32> = samples.iter().step_by(num_channels).copied().collect();
                    let right: Vec<f32> = if num_channels > 1 {
                        samples
                            .iter()
                            .skip(1)
                            .step_by(num_channels)
                            .copied()
                            .collect()
                    } else {
                        left.clone()
                    };
                    SourceStats {
                        pid: *pid,
                        left_rms_db: calculate_rms_db(&left),
                        right_rms_db: calculate_rms_db(&right),
                    }
                })
                .collect();

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
                    per_source_stats: per_source_stats.clone(),
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
                        per_source_stats,
                    }
                }
            };
            let _ = event_tx.send(CaptureEvent::StatsUpdate(stats));
            last_status_update = Instant::now();
        }
    }

    // Stop all streams
    for client in &clients {
        let _ = client.audio_client.stop_stream();
    }

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
