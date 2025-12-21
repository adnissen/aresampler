//! Color scheme constants for the application UI

use gpui::{Rgba, rgb};

// Backgrounds
pub fn bg_primary() -> Rgba {
    rgb(0x0a0a0b)
}
pub fn bg_secondary() -> Rgba {
    rgb(0x111113)
}
pub fn bg_tertiary() -> Rgba {
    rgb(0x18181b)
}

// Borders
pub fn border() -> Rgba {
    rgb(0x27272a)
}

// Text
pub fn text_primary() -> Rgba {
    rgb(0xfafafa)
}
pub fn text_secondary() -> Rgba {
    rgb(0xa1a1aa)
}
pub fn text_muted() -> Rgba {
    rgb(0x52525b)
}

// Accent
pub fn accent() -> Rgba {
    rgb(0x22d3ee)
}

// Recording
pub fn recording() -> Rgba {
    rgb(0xef4444)
}

// Success (for level display)
pub fn success() -> Rgba {
    rgb(0x22c55e)
}

// Error
pub fn error_bg() -> Rgba {
    rgb(0x5c1a1a)
}
pub fn error_text() -> Rgba {
    rgb(0xff6b6b)
}

// File icon purple
pub fn file_icon() -> Rgba {
    rgb(0x8b5cf6)
}
