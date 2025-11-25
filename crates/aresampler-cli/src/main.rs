use anyhow::{Context, Result};
use aresampler_core::{
    initialize_audio, process_exists, CaptureConfig, CaptureEvent, CaptureSession,
};
use clap::Parser;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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

    // Initialize COM for WASAPI
    initialize_audio()?;

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
        anyhow::bail!("Process with PID {} does not exist", args.pid);
    }

    println!("Starting audio capture for PID: {}", args.pid);
    println!("Output file: {}", args.output.display());
    println!("Press Ctrl+C to stop recording\n");

    // Create capture configuration
    let config = CaptureConfig {
        pid: args.pid,
        output_path: args.output.clone(),
        ..Default::default()
    };

    println!("Audio format:");
    println!("  Sample rate: {} Hz", config.sample_rate);
    println!("  Channels: {}", config.channels);
    println!("  Bits per sample: {}", config.bits_per_sample);
    println!();

    // Start capture session
    let mut session = CaptureSession::new(config);
    let event_rx = session.start()?;

    // Main loop - process events until Ctrl+C
    while running.load(Ordering::SeqCst) {
        // Check for events with a short timeout
        match event_rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(CaptureEvent::Started { buffer_size }) => {
                println!("Buffer size: {} frames", buffer_size);
                println!();
                println!("Recording started...\n");
            }
            Ok(CaptureEvent::StatsUpdate(stats)) => {
                print!(
                    "\rRecording: {:.1}s | Frames: {} | Size: {:.2} MB | Buffer: {} frames    ",
                    stats.duration_secs,
                    stats.total_frames,
                    stats.file_size_bytes as f64 / (1024.0 * 1024.0),
                    stats.buffer_frames
                );
                std::io::stdout().flush().ok();
            }
            Ok(CaptureEvent::Stopped) => {
                break;
            }
            Ok(CaptureEvent::Error(msg)) => {
                eprintln!("\nError: {}", msg);
                break;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // Continue checking running flag
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                break;
            }
        }
    }

    // Stop the session
    session.stop()?;

    println!("\n\nRecording complete!");
    println!("Output: {}", args.output.display());

    Ok(())
}
