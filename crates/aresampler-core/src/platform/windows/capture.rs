//! Windows audio capture implementation using WASAPI

use crate::process::process_exists;
use crate::types::{CaptureCommand, CaptureConfig, CaptureEvent, CaptureStats};
use anyhow::{anyhow, Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use std::collections::VecDeque;
use std::io::BufWriter;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use wasapi::{AudioClient, Direction, SampleType, StreamMode, WaveFormat};

/// Manages an audio capture session on Windows
pub struct CaptureSession {
    config: CaptureConfig,
    command_tx: Option<Sender<CaptureCommand>>,
    thread_handle: Option<JoinHandle<()>>,
}

impl CaptureSession {
    pub fn new(config: CaptureConfig) -> Self {
        Self {
            config,
            command_tx: None,
            thread_handle: None,
        }
    }

    /// Start capture in a background thread
    /// Returns a receiver for capture events
    pub fn start(&mut self) -> Result<Receiver<CaptureEvent>> {
        let (event_tx, event_rx) = channel();
        let (command_tx, command_rx) = channel();

        let config = self.config.clone();

        let handle = thread::spawn(move || {
            capture_thread_main(config, command_rx, event_tx);
        });

        self.command_tx = Some(command_tx);
        self.thread_handle = Some(handle);

        Ok(event_rx)
    }

    /// Stop the capture
    pub fn stop(&mut self) -> Result<()> {
        if let Some(tx) = self.command_tx.take() {
            let _ = tx.send(CaptureCommand::Stop);
        }
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
        Ok(())
    }

    /// Check if currently recording
    pub fn is_recording(&self) -> bool {
        self.command_tx.is_some()
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

fn run_capture(
    config: &CaptureConfig,
    command_rx: &Receiver<CaptureCommand>,
    event_tx: &Sender<CaptureEvent>,
) -> Result<()> {
    // Validate that the process exists
    if !process_exists(config.pid) {
        return Err(anyhow!("Process with PID {} does not exist", config.pid));
    }

    // Create application loopback client
    // Note: include_child_processes is now always true (hardcoded)
    let mut audio_client = AudioClient::new_application_loopback_client(config.pid, true)
        .map_err(|e| {
            anyhow!(
                "Failed to create application loopback client: {}. Make sure the PID is valid.",
                e
            )
        })?;

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

    audio_client
        .initialize_client(&desired_format, &Direction::Capture, &mode)
        .map_err(|e| anyhow!("Failed to initialize audio client: {}", e))?;

    // Get event handle
    let event_handle = audio_client
        .set_get_eventhandle()
        .map_err(|e| anyhow!("Failed to get event handle: {}", e))?;

    // Get capture client
    let capture_client = audio_client
        .get_audiocaptureclient()
        .map_err(|e| anyhow!("Failed to get audio capture client: {}", e))?;

    let buffer_size = audio_client.get_buffer_size().unwrap_or(4800);

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
    let mut wav_writer = WavWriter::new(buf_writer, wav_spec).context("Failed to create WAV writer")?;

    // Start the stream
    audio_client
        .start_stream()
        .map_err(|e| anyhow!("Failed to start audio stream: {}", e))?;

    let start_time = Instant::now();
    let mut total_frames: u64 = 0;
    let mut total_bytes: u64 = 0;
    let mut last_status_update = Instant::now();
    let mut audio_buffer: VecDeque<u8> = VecDeque::new();
    #[allow(unused_assignments)]
    let mut current_buffer_frames = 0usize;

    // Main capture loop
    loop {
        // Check for stop command (non-blocking)
        if let Ok(CaptureCommand::Stop) = command_rx.try_recv() {
            break;
        }

        // Wait for audio data (timeout 100ms)
        if event_handle.wait_for_event(100).is_err() {
            continue;
        }

        // Check how many frames are available
        let frames_available = match capture_client.get_next_packet_size() {
            Ok(Some(frames)) => frames,
            Ok(None) => continue,
            Err(_) => continue,
        };

        if frames_available == 0 {
            continue;
        }

        // Read available frames
        audio_buffer.clear();
        match capture_client.read_from_device_to_deque(&mut audio_buffer) {
            Ok(_buffer_flags) => {
                let num_bytes = audio_buffer.len();
                if num_bytes == 0 {
                    continue;
                }

                // Convert bytes to f32 samples and write to WAV
                let bytes = audio_buffer.make_contiguous();
                let samples = bytes_to_f32_samples(bytes);

                for &sample in samples {
                    wav_writer
                        .write_sample(sample)
                        .context("Failed to write sample to WAV")?;
                }

                let num_frames = num_bytes / block_align as usize;
                total_frames += num_frames as u64;
                total_bytes += num_bytes as u64;
                current_buffer_frames = num_frames;

                // Update status every 100ms
                if last_status_update.elapsed() >= Duration::from_millis(100) {
                    let duration = start_time.elapsed();
                    // Separate interleaved samples into channels and calculate RMS
                    let num_channels = config.channels as usize;
                    let left_samples: Vec<f32> =
                        samples.iter().step_by(num_channels).copied().collect();
                    let right_samples: Vec<f32> = if num_channels > 1 {
                        samples.iter().skip(1).step_by(num_channels).copied().collect()
                    } else {
                        left_samples.clone()
                    };
                    let left_rms_db = calculate_rms_db(&left_samples);
                    let right_rms_db = calculate_rms_db(&right_samples);
                    let stats = CaptureStats {
                        duration_secs: duration.as_secs_f64(),
                        total_frames,
                        file_size_bytes: total_bytes,
                        buffer_frames: current_buffer_frames,
                        is_recording: true,
                        left_rms_db,
                        right_rms_db,
                    };
                    let _ = event_tx.send(CaptureEvent::StatsUpdate(stats));
                    last_status_update = Instant::now();
                }
            }
            Err(e) => {
                let err_str = format!("{}", e);
                if err_str.contains("AUDCLNT_S_BUFFER_EMPTY") || err_str.contains("0x08890001") {
                    continue;
                }
                return Err(anyhow!("Error reading audio data: {}", e));
            }
        }
    }

    // Stop the stream
    audio_client
        .stop_stream()
        .map_err(|e| anyhow!("Failed to stop audio stream: {}", e))?;

    // Finalize the WAV file
    wav_writer.finalize().context("Failed to finalize WAV file")?;

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
