//! macOS-specific audio capture implementation using ScreenCaptureKit

mod audio;
mod capture;
mod icon;
mod permission;
mod session;

pub use audio::initialize_audio;
pub use capture::CaptureSession;
pub use icon::get_app_icon_png;
pub use permission::{is_capture_available, request_capture_permission};
pub use session::enumerate_audio_sessions;
