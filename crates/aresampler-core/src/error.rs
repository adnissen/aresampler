//! Error types for aresampler-core

use thiserror::Error;

/// Errors that can occur during audio capture
#[derive(Debug, Error)]
pub enum CaptureError {
    #[error("Process with PID {0} not found")]
    ProcessNotFound(u32),

    #[error("Permission denied for audio capture")]
    PermissionDenied,

    #[error("Failed to initialize audio subsystem: {0}")]
    InitializationFailed(String),

    #[error("Failed to create capture stream: {0}")]
    StreamCreationFailed(String),

    #[error("Failed to write audio file: {0}")]
    FileWriteError(#[from] std::io::Error),

    #[error("Audio capture error: {0}")]
    CaptureError(String),

    #[error("Session enumeration failed: {0}")]
    EnumerationFailed(String),
}
