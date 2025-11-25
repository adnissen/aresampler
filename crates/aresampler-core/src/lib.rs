pub mod capture;
pub mod process;
pub mod session;

pub use capture::{CaptureCommand, CaptureConfig, CaptureEvent, CaptureSession, CaptureStats};
pub use process::{get_process_info, process_exists, ProcessInfo};
pub use session::{enumerate_audio_sessions, AudioSessionInfo};

/// Initialize COM for WASAPI (must be called before any WASAPI operations)
pub fn initialize_audio() -> anyhow::Result<()> {
    let hr = wasapi::initialize_mta();
    if hr.is_err() {
        return Err(anyhow::anyhow!("Failed to initialize COM: HRESULT {:#x}", hr.0));
    }
    Ok(())
}
