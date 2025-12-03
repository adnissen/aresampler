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
    /// Application icon as PNG bytes (macOS only, None if unavailable)
    pub icon_png: Option<Vec<u8>>,
}

/// Configuration for audio capture
#[derive(Clone, Debug)]
pub struct CaptureConfig {
    /// Process IDs to capture audio from (can be multiple applications)
    pub pids: Vec<u32>,
    pub output_path: PathBuf,
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            pids: Vec::new(),
            output_path: PathBuf::new(),
            sample_rate: 48000,
            channels: 2,
            bits_per_sample: 32,
        }
    }
}

/// Configuration for monitoring mode (pre-recording with ring buffer)
#[derive(Clone, Debug)]
pub struct MonitorConfig {
    /// Process IDs to capture audio from (can be multiple applications)
    pub pids: Vec<u32>,
    pub sample_rate: u32,
    pub channels: u16,
    /// Pre-roll buffer duration in seconds (0 = no buffering)
    pub pre_roll_duration_secs: f32,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            pids: Vec::new(),
            sample_rate: 48000,
            channels: 2,
            pre_roll_duration_secs: 10.0,
        }
    }
}

/// Per-source audio statistics
#[derive(Clone, Debug, Default)]
pub struct SourceStats {
    /// Process ID this stat belongs to
    pub pid: u32,
    /// Left channel RMS level in dB (0 dB = full scale, -60 dB = silence)
    pub left_rms_db: f32,
    /// Right channel RMS level in dB (0 dB = full scale, -60 dB = silence)
    pub right_rms_db: f32,
}

/// Real-time statistics during capture
#[derive(Clone, Debug)]
pub struct CaptureStats {
    pub duration_secs: f64,
    pub total_frames: u64,
    pub file_size_bytes: u64,
    pub buffer_frames: usize,
    pub is_recording: bool,
    /// True when in monitoring mode (before recording starts)
    pub is_monitoring: bool,
    /// Current pre-roll buffer fill level in seconds
    pub pre_roll_buffer_secs: f32,
    /// Left channel RMS level in dB (0 dB = full scale, -60 dB = silence)
    pub left_rms_db: f32,
    /// Right channel RMS level in dB (0 dB = full scale, -60 dB = silence)
    pub right_rms_db: f32,
    /// Per-source volume levels (empty if unavailable, e.g., macOS multi-app)
    pub per_source_stats: Vec<SourceStats>,
}

impl Default for CaptureStats {
    fn default() -> Self {
        Self {
            duration_secs: 0.0,
            total_frames: 0,
            file_size_bytes: 0,
            buffer_frames: 0,
            is_recording: false,
            is_monitoring: false,
            pre_roll_buffer_secs: 0.0,
            left_rms_db: -60.0,
            right_rms_db: -60.0,
            per_source_stats: Vec::new(),
        }
    }
}

/// Commands sent to the capture thread
#[derive(Debug)]
pub enum CaptureCommand {
    Stop,
    /// Transition from monitoring to recording mode
    StartRecording {
        output_path: PathBuf,
    },
    /// Resize the pre-roll buffer while monitoring
    ResizePreRoll {
        duration_secs: f32,
    },
}

/// Events sent from the capture thread
#[derive(Debug)]
pub enum CaptureEvent {
    Started {
        buffer_size: usize,
    },
    StatsUpdate(CaptureStats),
    Stopped,
    Error(String),
    /// Monitoring mode has started (capturing to ring buffer)
    MonitoringStarted,
    /// Recording has started (pre-roll buffer was written to file)
    RecordingStarted {
        /// How many seconds of pre-roll audio were captured
        pre_roll_secs: f32,
    },
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

    /// Start monitoring mode - captures audio to ring buffer without recording
    fn start_monitoring(&mut self, config: MonitorConfig)
        -> anyhow::Result<Receiver<CaptureEvent>>;

    /// Transition from monitoring to recording
    fn start_recording(&mut self, output_path: PathBuf) -> anyhow::Result<()>;

    /// Check if currently in monitoring mode
    fn is_monitoring(&self) -> bool;
}
