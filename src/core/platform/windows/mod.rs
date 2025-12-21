//! Windows-specific audio capture implementation using WASAPI

mod audio;
mod capture;
mod icon;
mod input_device;
mod session;

pub use audio::{initialize_audio, is_capture_available, request_capture_permission};
pub use capture::CaptureSession;
pub use icon::get_app_icon_png;
pub use input_device::enumerate_input_devices;
pub use session::enumerate_audio_sessions;
