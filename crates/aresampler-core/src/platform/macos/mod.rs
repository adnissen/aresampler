//! macOS-specific audio capture implementation using ScreenCaptureKit

mod audio;
mod capture;
mod icon;
mod permission;
mod session;

pub use audio::initialize_audio;
pub use capture::CaptureSession;
pub use permission::{is_capture_available, request_capture_permission};
pub use session::enumerate_audio_sessions;
