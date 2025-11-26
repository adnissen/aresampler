//! Shared types for aresampler-core
//!
//! Platform-agnostic types used across the library.

use std::path::PathBuf;
use std::sync::mpsc::Receiver;

/// Permission status for audio capture (relevant on macOS)
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PermissionStatus {
    Granted,
    Denied,
    Unknown,
}

/// Information about an active audio session/application
#[derive(Clone, Debug)]
pub struct AudioSessionInfo {
    pub pid: u32,
    pub name: String,
    /// Bundle identifier (macOS only, None on Windows)
    pub bundle_id: Option<String>,
}

/// Configuration for audio capture
#[derive(Clone, Debug)]
pub struct CaptureConfig {
    pub pid: u32,
    pub output_path: PathBuf,
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            pid: 0,
            output_path: PathBuf::new(),
            sample_rate: 48000,
            channels: 2,
            bits_per_sample: 32,
        }
    }
}

/// Real-time statistics during capture
#[derive(Clone, Debug, Default)]
pub struct CaptureStats {
    pub duration_secs: f64,
    pub total_frames: u64,
    pub file_size_bytes: u64,
    pub buffer_frames: usize,
    pub is_recording: bool,
}

/// Commands sent to the capture thread
#[derive(Debug)]
pub enum CaptureCommand {
    Stop,
}

/// Events sent from the capture thread
#[derive(Debug)]
pub enum CaptureEvent {
    Started { buffer_size: usize },
    StatsUpdate(CaptureStats),
    Stopped,
    Error(String),
}

/// Process information
#[derive(Clone, Debug)]
pub struct ProcessInfo {
    pub pid: u32,
    pub name: String,
    pub exe_path: Option<String>,
}

/// Trait for platform-specific capture session implementation
pub trait CaptureSessionImpl: Send {
    fn start(&mut self) -> anyhow::Result<Receiver<CaptureEvent>>;
    fn stop(&mut self) -> anyhow::Result<()>;
    fn is_recording(&self) -> bool;
}
