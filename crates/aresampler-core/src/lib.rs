//! aresampler-core - Cross-platform process audio capture
//!
//! This library provides a platform-agnostic API for capturing audio from
//! specific processes. It supports:
//! - Windows: via WASAPI application loopback
//! - macOS: via ScreenCaptureKit
//!
//! # Example
//!
//! ```no_run
//! use aresampler_core::{
//!     initialize_audio, enumerate_audio_sessions,
//!     CaptureConfig, CaptureSession, CaptureEvent,
//! };
//! use std::path::PathBuf;
//!
//! // Initialize the audio subsystem
//! initialize_audio().expect("Failed to initialize audio");
//!
//! // List available audio sessions
//! let sessions = enumerate_audio_sessions().expect("Failed to enumerate sessions");
//! for session in &sessions {
//!     println!("Found: {} (PID: {})", session.name, session.pid);
//! }
//!
//! // Configure and start capture
//! if let Some(session) = sessions.first() {
//!     let config = CaptureConfig {
//!         pid: session.pid,
//!         output_path: PathBuf::from("output.wav"),
//!         ..Default::default()
//!     };
//!
//!     let mut capture = CaptureSession::new(config);
//!     let events = capture.start().expect("Failed to start capture");
//!
//!     // Handle events...
//! }
//! ```

pub mod error;
mod platform;
pub mod process;
pub mod ring_buffer;
pub mod types;

// Re-export all public types from types module
pub use types::{
    AudioSessionInfo, CaptureCommand, CaptureConfig, CaptureEvent, CaptureStats, MonitorConfig,
    PermissionStatus, ProcessInfo, SourceStats,
};

// Re-export ring buffer
pub use ring_buffer::AudioRingBuffer;

// Re-export process utilities
pub use process::{get_parent_pid, get_process_info, process_exists};

// Re-export platform-specific implementations through a unified API
pub use platform::{
    enumerate_audio_sessions, initialize_audio, is_capture_available, request_capture_permission,
    CaptureSession,
};

// Re-export error types
pub use error::CaptureError;
