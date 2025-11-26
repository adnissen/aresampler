//! Platform-specific implementations
//!
//! This module provides platform-specific audio capture implementations
//! while exposing a unified API.

#[cfg(target_os = "windows")]
pub mod windows;
#[cfg(target_os = "windows")]
pub use windows::*;

#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(target_os = "macos")]
pub use macos::*;

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
compile_error!("aresampler-core only supports Windows and macOS");
