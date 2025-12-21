//! Windows audio initialization

use crate::core::types::PermissionStatus;
use anyhow::Result;

/// Initialize COM for WASAPI (must be called before any WASAPI operations)
pub fn initialize_audio() -> Result<()> {
    let hr = wasapi::initialize_mta();
    if hr.is_err() {
        return Err(anyhow::anyhow!(
            "Failed to initialize COM: HRESULT {:#x}",
            hr.0
        ));
    }
    Ok(())
}

/// Check if audio capture is available on Windows
///
/// Always returns true if COM is initialized successfully.
pub fn is_capture_available() -> Result<bool> {
    // On Windows, capture is always available if COM initializes
    Ok(true)
}

/// Request capture permission on Windows
///
/// On Windows, no permission is needed - always returns Granted.
pub fn request_capture_permission() -> Result<PermissionStatus> {
    Ok(PermissionStatus::Granted)
}
