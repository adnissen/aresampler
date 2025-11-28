use anyhow::{Context, Result};
use aresampler_core::{
    initialize_audio, is_capture_available, process_exists, request_capture_permission,
    CaptureConfig, CaptureEvent, CaptureSession, MonitorConfig, PermissionStatus,
};
use clap::Parser;
use std::io::{BufRead, Write};
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

    /// Enable monitoring mode - start capturing to a ring buffer immediately,
    /// then press Enter to start recording. The pre-roll buffer will be
    /// prepended to the recording.
    #[arg(long)]
    monitor: bool,

    /// Pre-roll buffer duration in seconds (0 = disabled, default 10).
    /// Only used with --monitor flag.
    #[arg(long, default_value = "10.0")]
    pre_roll: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize audio subsystem
    initialize_audio()?;

    // Check if capture is available (permission check on macOS)
    if !is_capture_available()? {
        eprintln!("Audio capture is not available.");
        eprintln!("Requesting capture permission...");

        match request_capture_permission()? {
            PermissionStatus::Granted => {
                println!("Permission granted!");
            }
            PermissionStatus::Denied => {
                eprintln!("Permission denied. Please grant Screen Recording permission in:");
                eprintln!("  System Preferences > Privacy & Security > Screen Recording");
                std::process::exit(1);
            }
            PermissionStatus::Unknown => {
                eprintln!("Permission status unknown. Please check system settings.");
                std::process::exit(1);
            }
        }
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
        anyhow::bail!("Process with PID {} does not exist", args.pid);
    }

    println!("Starting audio capture for PID: {}", args.pid);
    println!("Output file: {}", args.output.display());
    println!("Press Ctrl+C to stop\n");

    if args.monitor {
        // Monitor mode - start with ring buffer, transition on Enter
        run_monitor_mode(&args, running)?;
    } else {
        // Direct recording mode (legacy behavior)
        run_direct_mode(&args, running)?;
    }

    Ok(())
}

/// Run in direct recording mode (original behavior)
fn run_direct_mode(args: &Args, running: Arc<AtomicBool>) -> Result<()> {
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
            Ok(CaptureEvent::MonitoringStarted) | Ok(CaptureEvent::RecordingStarted { .. }) => {
                // Not expected in direct mode
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

/// Run in monitor mode with pre-roll buffer
fn run_monitor_mode(args: &Args, running: Arc<AtomicBool>) -> Result<()> {
    let monitor_config = MonitorConfig {
        pid: args.pid,
        pre_roll_duration_secs: args.pre_roll,
        ..Default::default()
    };

    println!("Monitor mode enabled");
    println!("  Pre-roll buffer: {:.1} seconds", args.pre_roll);
    println!("  Sample rate: {} Hz", monitor_config.sample_rate);
    println!("  Channels: {}", monitor_config.channels);
    println!();

    // Start monitoring session
    let mut session = CaptureSession::new_empty();
    let event_rx = session.start_monitoring(monitor_config)?;

    // Spawn a thread to read Enter key
    let enter_pressed = Arc::new(AtomicBool::new(false));
    let enter_pressed_clone = enter_pressed.clone();
    let running_clone = running.clone();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut stdin_lock = stdin.lock();
        let mut line = String::new();
        while running_clone.load(Ordering::SeqCst) && !enter_pressed_clone.load(Ordering::SeqCst) {
            if stdin_lock.read_line(&mut line).is_ok() {
                enter_pressed_clone.store(true, Ordering::SeqCst);
                break;
            }
        }
    });

    let mut is_recording = false;

    // Main loop
    while running.load(Ordering::SeqCst) {
        // Check if Enter was pressed and we should start recording
        if !is_recording && enter_pressed.load(Ordering::SeqCst) {
            println!("\nStarting recording...");
            session.start_recording(args.output.clone())?;
            is_recording = true;
        }

        // Check for events with a short timeout
        match event_rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(CaptureEvent::MonitoringStarted) => {
                println!("Monitoring started. Press Enter to start recording...\n");
            }
            Ok(CaptureEvent::RecordingStarted { pre_roll_secs }) => {
                println!("Recording started (pre-roll: {:.1}s)\n", pre_roll_secs);
            }
            Ok(CaptureEvent::StatsUpdate(stats)) => {
                if stats.is_monitoring {
                    print!(
                        "\rMonitoring: Buffer {:.1}s/{:.1}s | L: {:.1} dB | R: {:.1} dB    ",
                        stats.pre_roll_buffer_secs,
                        args.pre_roll,
                        stats.left_rms_db,
                        stats.right_rms_db
                    );
                } else if stats.is_recording {
                    print!(
                        "\rRecording: {:.1}s | Frames: {} | Size: {:.2} MB    ",
                        stats.duration_secs,
                        stats.total_frames,
                        stats.file_size_bytes as f64 / (1024.0 * 1024.0)
                    );
                }
                std::io::stdout().flush().ok();
            }
            Ok(CaptureEvent::Stopped) => {
                break;
            }
            Ok(CaptureEvent::Error(msg)) => {
                eprintln!("\nError: {}", msg);
                break;
            }
            Ok(CaptureEvent::Started { .. }) => {
                // Not expected in monitor mode
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

    if is_recording {
        println!("\n\nRecording complete!");
        println!("Output: {}", args.output.display());
    } else {
        println!("\n\nMonitoring stopped (no recording saved)");
    }

    Ok(())
}
