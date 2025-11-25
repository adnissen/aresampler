use anyhow::{anyhow, Context, Result};
use clap::Parser;
use hound::{SampleFormat, WavSpec, WavWriter};
use std::collections::VecDeque;
use std::io::BufWriter;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use wasapi::{AudioClient, Direction, SampleType, StreamMode, WaveFormat};

/// Check if a process with the given PID exists
fn process_exists(pid: u32) -> bool {
    // PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;

    #[link(name = "kernel32")]
    extern "system" {
        fn OpenProcess(dwDesiredAccess: u32, bInheritHandle: i32, dwProcessId: u32) -> *mut std::ffi::c_void;
        fn CloseHandle(hObject: *mut std::ffi::c_void) -> i32;
    }

    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            false
        } else {
            CloseHandle(handle);
            true
        }
    }
}

#[derive(Parser)]
#[command(name = "aresampler")]
#[command(about = "Record audio from a specific application to a WAV file")]
struct Args {
    /// Process ID of the application to record from
    #[arg(short, long)]
    pid: u32,

    /// Output WAV file path
    #[arg(short, long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize COM for multi-threaded apartment (required for WASAPI)
    let hr = wasapi::initialize_mta();
    if hr.is_err() {
        return Err(anyhow!("Failed to initialize COM: HRESULT {:#x}", hr.0));
    }

    // Set up Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        eprintln!("\nReceived Ctrl+C, stopping recording...");
        r.store(false, Ordering::SeqCst);
    })
    .context("Failed to set Ctrl+C handler")?;

    // Validate that the process exists
    if !process_exists(args.pid) {
        return Err(anyhow!("Process with PID {} does not exist", args.pid));
    }

    println!("Starting audio capture for PID: {}", args.pid);
    println!("Output file: {}", args.output.display());
    println!("Press Ctrl+C to stop recording\n");

    // Create application loopback client
    // include_tree = true means we capture from the process and all its children
    let mut audio_client = AudioClient::new_application_loopback_client(args.pid, true)
        .map_err(|e| anyhow!("Failed to create application loopback client: {}. Make sure the PID is valid and the process exists.", e))?;

    // Define the desired audio format
    // 32-bit float, 48kHz, stereo - standard high-quality format
    let sample_rate = 48000u32;
    let channels = 2u16;
    let bits_per_sample = 32u16;
    let block_align = (channels * bits_per_sample / 8) as u16; // 8 bytes per frame for stereo float32

    let desired_format = WaveFormat::new(
        bits_per_sample as usize,      // bits per sample
        bits_per_sample as usize,      // valid bits per sample
        &SampleType::Float,            // sample type
        sample_rate as usize,          // sample rate
        channels as usize,             // channels
        None,                          // channel mask (None = default)
    );

    println!("Audio format:");
    println!("  Sample rate: {} Hz", sample_rate);
    println!("  Channels: {}", channels);
    println!("  Bits per sample: {}", bits_per_sample);
    println!("  Block align: {} bytes", block_align);
    println!();

    // Initialize the audio client for capture with autoconvert enabled
    // Buffer duration in 100-nanosecond units (200ms = 2_000_000)
    let buffer_duration_hns = 2_000_000i64;
    let mode = StreamMode::EventsShared {
        autoconvert: true,
        buffer_duration_hns,
    };

    audio_client
        .initialize_client(&desired_format, &Direction::Capture, &mode)
        .map_err(|e| anyhow!("Failed to initialize audio client: {}", e))?;

    // Get event handle for event-driven capture
    let event_handle = audio_client
        .set_get_eventhandle()
        .map_err(|e| anyhow!("Failed to get event handle: {}", e))?;

    // Get the capture client
    let capture_client = audio_client
        .get_audiocaptureclient()
        .map_err(|e| anyhow!("Failed to get audio capture client: {}", e))?;

    // Try to get buffer size (may not work for loopback, so use a default)
    let buffer_size = audio_client.get_buffer_size().unwrap_or(4800);
    println!("Buffer size: {} frames", buffer_size);
    println!();

    // Set up WAV writer with 32-bit float format
    let wav_spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let file = std::fs::File::create(&args.output)
        .with_context(|| format!("Failed to create output file: {}", args.output.display()))?;
    let buf_writer = BufWriter::new(file);
    let mut wav_writer = WavWriter::new(buf_writer, wav_spec)
        .context("Failed to create WAV writer")?;

    // Start the stream
    audio_client
        .start_stream()
        .map_err(|e| anyhow!("Failed to start audio stream: {}", e))?;

    println!("Recording started...\n");

    let start_time = Instant::now();
    let mut total_frames: u64 = 0;
    let mut total_bytes: u64 = 0;
    let mut last_status_update = Instant::now();
    let mut audio_buffer: VecDeque<u8> = VecDeque::new();

    // Main capture loop
    while running.load(Ordering::SeqCst) {
        // Wait for audio data (timeout 100ms)
        if event_handle.wait_for_event(100).is_err() {
            // Timeout - no data available yet, continue checking shutdown flag
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

        // Read available frames into the deque
        audio_buffer.clear();
        match capture_client.read_from_device_to_deque(&mut audio_buffer) {
            Ok(_buffer_flags) => {
                let num_bytes = audio_buffer.len();
                if num_bytes == 0 {
                    continue;
                }

                // Convert bytes to f32 samples and write to WAV
                // Make the deque contiguous and cast to f32
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

                // Update status every 500ms
                if last_status_update.elapsed() >= Duration::from_millis(500) {
                    let duration = start_time.elapsed();
                    let duration_secs = duration.as_secs_f64();
                    print!(
                        "\rRecording: {:.1}s | Frames: {} | Size: {:.2} MB | Buffer: {} frames    ",
                        duration_secs,
                        total_frames,
                        total_bytes as f64 / (1024.0 * 1024.0),
                        num_frames
                    );
                    use std::io::Write;
                    std::io::stdout().flush().ok();
                    last_status_update = Instant::now();
                }
            }
            Err(e) => {
                // Check if this is a "no data" situation or a real error
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

    let duration = start_time.elapsed();
    println!("\n\nRecording complete!");
    println!("Duration: {:.2} seconds", duration.as_secs_f64());
    println!("Total frames: {}", total_frames);
    println!("File size: {:.2} MB", total_bytes as f64 / (1024.0 * 1024.0));
    println!("Output: {}", args.output.display());

    Ok(())
}

/// Safely cast a byte slice to a slice of f32
fn bytes_to_f32_samples(bytes: &[u8]) -> &[f32] {
    let len = bytes.len() / 4;
    if len == 0 || bytes.len() % 4 != 0 {
        return &[];
    }
    // Check alignment
    if bytes.as_ptr() as usize % std::mem::align_of::<f32>() != 0 {
        // Misaligned data - should not happen with WASAPI but handle gracefully
        return &[];
    }
    // Safety: We trust that WASAPI gives us properly aligned f32 data
    // when the format is 32-bit float
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len) }
}
