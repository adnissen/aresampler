//! Windows-specific audio capture implementation using WASAPI

mod audio;
mod capture;
mod icon;
mod session;

pub use audio::{initialize_audio, is_capture_available, request_capture_permission};
pub use capture::CaptureSession;
pub use session::enumerate_audio_sessions;
