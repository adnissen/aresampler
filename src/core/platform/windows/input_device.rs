//! Windows audio input device enumeration (stub)
//!
//! Microphone input is not yet implemented on Windows.
//! This module returns an empty list.

use crate::core::types::InputDevice;
use anyhow::Result;

/// Enumerate all available audio input devices (microphones).
///
/// Not implemented on Windows - always returns an empty list.
pub fn enumerate_input_devices() -> Result<Vec<InputDevice>> {
    Ok(Vec::new())
}
