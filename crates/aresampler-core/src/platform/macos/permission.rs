//! macOS TCC permission handling for ScreenCaptureKit

use crate::types::PermissionStatus;
use anyhow::Result;
use screencapturekit::shareable_content::SCShareableContent;

/// Check if audio capture is available on macOS
///
/// This checks if the "Screen Recording" permission has been granted.
/// ScreenCaptureKit requires this permission to capture application audio.
pub fn is_capture_available() -> Result<bool> {
    // Attempting to get shareable content will succeed only if we have permission
    match SCShareableContent::get() {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// Request capture permission on macOS
///
/// This triggers the TCC permission dialog for Screen Recording.
/// The user must grant permission in System Preferences > Privacy & Security.
pub fn request_capture_permission() -> Result<PermissionStatus> {
    // Querying SCShareableContent triggers the permission dialog
    match SCShareableContent::get() {
        Ok(_) => Ok(PermissionStatus::Granted),
        Err(_) => {
            // Permission was denied or not yet granted
            // We can't distinguish between these states easily
            Ok(PermissionStatus::Denied)
        }
    }
}
