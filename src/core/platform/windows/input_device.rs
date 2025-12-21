//! Windows audio input device enumeration using WASAPI

use crate::core::types::InputDevice;
use anyhow::{Context, Result};
use wasapi::{DeviceCollection, Direction};

/// Enumerate all available audio input devices (microphones).
///
/// Uses WASAPI device enumeration to list all capture devices.
pub fn enumerate_input_devices() -> Result<Vec<InputDevice>> {
    // Get default capture device ID for comparison
    let default_id = wasapi::get_default_device(&Direction::Capture)
        .ok()
        .and_then(|d| d.get_id().ok());

    // Get all capture devices
    let collection = DeviceCollection::new(&Direction::Capture)
        .context("Failed to enumerate capture devices")?;

    let mut devices = Vec::new();

    for device_result in &collection {
        let device = match device_result {
            Ok(d) => d,
            Err(_) => continue,
        };

        let id = match device.get_id() {
            Ok(id) => id,
            Err(_) => continue, // Skip devices we can't get an ID for
        };

        let name = match device.get_friendlyname() {
            Ok(name) => name,
            Err(_) => continue, // Skip devices we can't get a name for
        };

        let is_default = default_id.as_ref() == Some(&id);

        devices.push(InputDevice { id, name, is_default });
    }

    Ok(devices)
}
