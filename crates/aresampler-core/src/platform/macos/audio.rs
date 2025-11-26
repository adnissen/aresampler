//! macOS audio initialization

use anyhow::Result;

/// Initialize audio subsystem on macOS
///
/// On macOS, no special initialization is required.
/// This is a no-op for API compatibility with Windows.
pub fn initialize_audio() -> Result<()> {
    // No initialization needed on macOS
    Ok(())
}
