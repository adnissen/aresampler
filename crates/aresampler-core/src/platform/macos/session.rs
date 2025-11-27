//! macOS audio session enumeration using ScreenCaptureKit

use super::icon::get_app_icon_png;
use crate::types::AudioSessionInfo;
use anyhow::{anyhow, Result};
use screencapturekit::shareable_content::SCShareableContent;

/// Enumerate all applications that can be captured on macOS
///
/// Returns a list of running applications that can potentially produce audio.
/// Note: Unlike Windows WASAPI which only shows apps actively using audio,
/// ScreenCaptureKit returns all capturable applications.
pub fn enumerate_audio_sessions() -> Result<Vec<AudioSessionInfo>> {
    let content = SCShareableContent::get().map_err(|e| {
        anyhow!(
            "Failed to get shareable content. Screen Recording permission may be required: {:?}",
            e
        )
    })?;

    let mut sessions: Vec<AudioSessionInfo> = content
        .applications()
        .iter()
        .filter_map(|app| {
            // Filter out apps without a name
            let name = app.application_name();
            if name.is_empty() {
                return None;
            }

            let bundle_id = app.bundle_identifier();
            // Fetch icon using bundle_id
            let icon_png = get_app_icon_png(&bundle_id);

            Some(AudioSessionInfo {
                pid: app.process_id() as u32,
                name,
                bundle_id: Some(bundle_id),
                icon_png,
            })
        })
        .collect();

    // Sort by name for consistent display
    sessions.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));

    Ok(sessions)
}
