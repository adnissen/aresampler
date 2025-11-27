//! macOS audio capture implementation using ScreenCaptureKit

use crate::process::process_exists;
use crate::types::{CaptureCommand, CaptureConfig, CaptureEvent, CaptureStats};
use anyhow::{anyhow, Context, Result};
use core_media_rs::cm_sample_buffer::CMSampleBuffer;
use hound::{SampleFormat, WavSpec, WavWriter};
use screencapturekit::shareable_content::SCShareableContent;
use screencapturekit::stream::configuration::SCStreamConfiguration;
use screencapturekit::stream::content_filter::SCContentFilter;
use screencapturekit::stream::output_trait::SCStreamOutputTrait;
use screencapturekit::stream::output_type::SCStreamOutputType;
use screencapturekit::stream::SCStream;
use std::io::BufWriter;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Manages an audio capture session on macOS
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
    if let Err(e) = run_capture(&config, &command_rx, &event_tx) {
        let _ = event_tx.send(CaptureEvent::Error(e.to_string()));
    }

    let _ = event_tx.send(CaptureEvent::Stopped);
}

/// Shared state for the audio output handler
struct AudioOutputState {
    wav_writer: WavWriter<BufWriter<std::fs::File>>,
    total_frames: u64,
    total_bytes: u64,
    start_time: Instant,
    last_status_update: Instant,
    event_tx: Sender<CaptureEvent>,
}

/// Handler for receiving audio samples from ScreenCaptureKit
struct AudioHandler {
    state: Arc<Mutex<AudioOutputState>>,
    stop_flag: Arc<AtomicBool>,
}

impl SCStreamOutputTrait for AudioHandler {
    fn did_output_sample_buffer(&self, sample_buffer: CMSampleBuffer, of_type: SCStreamOutputType) {
        if of_type != SCStreamOutputType::Audio {
            return;
        }

        if self.stop_flag.load(Ordering::Relaxed) {
            return;
        }

        // Get audio buffer from sample buffer
        let Ok(audio_buffers) = sample_buffer.get_audio_buffer_list() else {
            return;
        };

        let mut state = match self.state.lock() {
            Ok(s) => s,
            Err(_) => return,
        };

        // ScreenCaptureKit provides non-interleaved (planar) audio:
        // Each buffer contains samples for a single channel.
        // We need to interleave them for WAV output.
        let buffers = audio_buffers.buffers();

        if buffers.is_empty() {
            return;
        }

        // Get samples from each channel buffer
        let channel_samples: Vec<&[f32]> = buffers
            .iter()
            .map(|b| bytes_to_f32_samples(b.data()))
            .collect();

        // All channels should have the same number of samples
        let num_frames = channel_samples.first().map(|s| s.len()).unwrap_or(0);
        if num_frames == 0 {
            return;
        }

        let num_channels = channel_samples.len();

        // Interleave samples: L0, R0, L1, R1, L2, R2, ...
        for frame_idx in 0..num_frames {
            for ch in 0..num_channels {
                if let Some(&sample) = channel_samples.get(ch).and_then(|s| s.get(frame_idx)) {
                    if state.wav_writer.write_sample(sample).is_err() {
                        return;
                    }
                }
            }
        }

        let num_bytes = buffers.iter().map(|b| b.data().len()).sum::<usize>();
        state.total_frames += num_frames as u64;
        state.total_bytes += num_bytes as u64;

        // Update status every 100ms
        if state.last_status_update.elapsed() >= Duration::from_millis(100) {
            let duration = state.start_time.elapsed();
            // Calculate RMS for each channel (left = 0, right = 1)
            let left_rms_db = channel_samples
                .first()
                .map(|s| calculate_rms_db(s))
                .unwrap_or(-60.0);
            let right_rms_db = channel_samples
                .get(1)
                .map(|s| calculate_rms_db(s))
                .unwrap_or(-60.0);
            let stats = CaptureStats {
                duration_secs: duration.as_secs_f64(),
                total_frames: state.total_frames,
                file_size_bytes: state.total_bytes,
                buffer_frames: num_frames,
                is_recording: true,
                left_rms_db,
                right_rms_db,
            };
            let _ = state.event_tx.send(CaptureEvent::StatsUpdate(stats));
            state.last_status_update = Instant::now();
        }
    }
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

    // Get shareable content
    let content = SCShareableContent::get()
        .map_err(|e| anyhow!("Failed to get shareable content: {:?}", e))?;

    // Find the application by PID
    let app = content
        .applications()
        .into_iter()
        .find(|a| a.process_id() as u32 == config.pid)
        .ok_or_else(|| {
            anyhow!(
                "Application with PID {} not found in shareable content",
                config.pid
            )
        })?;

    // Find a window belonging to this application to use for the content filter
    let windows = content.windows();
    let app_window = windows
        .iter()
        .find(|w| w.owning_application().process_id() == app.process_id());

    // Get the first display for fallback
    let displays = content.displays();
    let display = displays.first().ok_or_else(|| anyhow!("No displays found"))?;

    // Create content filter - prefer app window, fallback to display with app filter
    let filter = if let Some(window) = app_window {
        SCContentFilter::new().with_desktop_independent_window(window)
    } else {
        // Fallback: capture from display including only this application
        SCContentFilter::new()
            .with_display_including_application_excepting_windows(display, &[&app], &[])
    };

    // Configure the stream for audio capture
    let stream_config = SCStreamConfiguration::new()
        .set_captures_audio(true)
        .map_err(|e| anyhow!("Failed to enable audio capture: {:?}", e))?
        .set_sample_rate(config.sample_rate)
        .map_err(|e| anyhow!("Failed to set sample rate: {:?}", e))?
        .set_channel_count(config.channels as u8)
        .map_err(|e| anyhow!("Failed to set channel count: {:?}", e))?
        // Minimal video settings (required by API but we only want audio)
        .set_width(1)
        .map_err(|e| anyhow!("Failed to set width: {:?}", e))?
        .set_height(1)
        .map_err(|e| anyhow!("Failed to set height: {:?}", e))?;

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
    let wav_writer =
        WavWriter::new(buf_writer, wav_spec).context("Failed to create WAV writer")?;

    // Create shared state for the output handler
    let state = Arc::new(Mutex::new(AudioOutputState {
        wav_writer,
        total_frames: 0,
        total_bytes: 0,
        start_time: Instant::now(),
        last_status_update: Instant::now(),
        event_tx: event_tx.clone(),
    }));

    let stop_flag = Arc::new(AtomicBool::new(false));

    // Create the stream with output handler
    let audio_handler = AudioHandler {
        state: state.clone(),
        stop_flag: stop_flag.clone(),
    };

    let mut stream = SCStream::new(&filter, &stream_config);
    stream.add_output_handler(audio_handler, SCStreamOutputType::Audio);

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
