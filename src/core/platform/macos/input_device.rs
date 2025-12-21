//! macOS audio input device enumeration using ScreenCaptureKit's AVFoundation wrapper

use crate::core::types::InputDevice;
use anyhow::Result;
use screencapturekit::audio_devices::AudioInputDevice;

/// Enumerate all available audio input devices (microphones).
///
/// Uses ScreenCaptureKit's AVFoundation-based audio device enumeration.
/// Requires macOS 15.0+ for microphone capture to work with SCStream.
pub fn enumerate_input_devices() -> Result<Vec<InputDevice>> {
    Ok(AudioInputDevice::list()
        .into_iter()
        .map(|d| InputDevice {
            id: d.id,
            name: d.name,
            is_default: d.is_default,
        })
        .collect())
}
