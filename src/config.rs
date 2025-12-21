//! Configuration persistence for aresampler.
//!
//! This module handles loading and saving user preferences to a config file.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// User configuration that persists across sessions.
#[derive(Serialize, Deserialize, Default)]
pub struct Config {
    /// The name of the selected theme.
    pub theme_name: Option<String>,
}

impl Config {
    /// Get the config file path for the current platform.
    ///
    /// - macOS/Linux: `~/.config/aresampler/settings.json`
    /// - Windows: `%APPDATA%/aresampler/settings.json`
    fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|p| p.join("aresampler").join("settings.json"))
    }

    /// Load config from disk, returns default if not found or on error.
    pub fn load() -> Self {
        Self::config_path()
            .and_then(|path| fs::read_to_string(path).ok())
            .and_then(|contents| serde_json::from_str(&contents).ok())
            .unwrap_or_default()
    }

    /// Save config to disk.
    pub fn save(&self) -> Result<(), std::io::Error> {
        if let Some(path) = Self::config_path() {
            // Create the config directory if it doesn't exist
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let contents = serde_json::to_string_pretty(self)?;
            fs::write(path, contents)?;
        }
        Ok(())
    }
}
