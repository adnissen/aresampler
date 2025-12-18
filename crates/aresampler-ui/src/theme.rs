//! Theme integration with gpui-component
//!
//! This module loads the One Dark theme and provides helper functions
//! for accessing theme colors throughout the application.

use gpui::{App, Hsla};
use gpui_component::{Theme, ThemeSet};
use std::rc::Rc;

/// One Dark Theme JSON embedded in the binary
const ONE_DARK_THEME: &str = include_str!("../assets/themes/one-dark-theme.json");

/// Initialize the One Dark theme as the default theme for the application.
/// Call this after `gpui_component::init(cx)` in your main function.
pub fn init(cx: &mut App) {
    // Parse the One Dark theme
    let theme_set: ThemeSet =
        serde_json::from_str(ONE_DARK_THEME).expect("Failed to parse One Dark theme JSON");

    // Get the first theme config (One Dark)
    let one_dark_config = theme_set
        .themes
        .into_iter()
        .next()
        .expect("One Dark theme set should have at least one theme");

    // Get the global theme and set our One Dark as the dark theme
    let theme = Theme::global_mut(cx);
    let config_rc = Rc::new(one_dark_config);
    theme.dark_theme = config_rc.clone();

    // Apply the configuration
    theme.apply_config(&config_rc);

    // Disable shadows globally (helps with context menu shadow artifacts)
    theme.shadow = false;
}

/// Helper trait extension for accessing common theme colors.
/// This provides a more ergonomic API for the most commonly used colors.
pub trait OneDarkExt {
    /// Get the theme reference
    fn theme(&self) -> &Theme;

    // === BACKGROUND COLORS ===

    /// Main editor/content area background
    fn editor_background(&self) -> Hsla {
        self.theme().background
    }

    /// Surface background for panels and panes
    fn surface_background(&self) -> Hsla {
        self.theme().sidebar
    }

    /// Elevated surface background
    fn elevated_surface_background(&self) -> Hsla {
        self.theme().popover
    }

    /// UI element default background
    fn element_background(&self) -> Hsla {
        self.theme().secondary
    }

    /// Hovered element background
    fn element_hover(&self) -> Hsla {
        self.theme().accent
    }

    /// Active/pressed element background
    fn element_active(&self) -> Hsla {
        self.theme().secondary_active
    }

    /// Selected element background
    fn element_selected(&self) -> Hsla {
        self.theme().secondary_active
    }

    /// Disabled element background
    fn element_disabled(&self) -> Hsla {
        self.theme().muted
    }

    // === TEXT COLORS ===

    /// Primary text color
    fn text(&self) -> Hsla {
        self.theme().foreground
    }

    /// De-emphasized/muted text
    fn text_muted(&self) -> Hsla {
        self.theme().muted_foreground
    }

    /// Placeholder text
    fn text_placeholder(&self) -> Hsla {
        self.theme().muted_foreground
    }

    /// Disabled text
    fn text_disabled(&self) -> Hsla {
        self.theme().muted_foreground
    }

    /// Accent text for links and highlights
    fn text_accent(&self) -> Hsla {
        self.theme().link
    }

    // === BORDER COLORS ===

    /// Standard border color
    fn border(&self) -> Hsla {
        self.theme().border
    }

    /// Variant/subtle border color
    fn border_variant(&self) -> Hsla {
        self.theme().sidebar_border
    }

    /// Focused border color
    fn border_focused(&self) -> Hsla {
        self.theme().ring
    }

    /// Selected border color
    fn border_selected(&self) -> Hsla {
        self.theme().list_active_border
    }

    /// Disabled border color
    fn border_disabled(&self) -> Hsla {
        self.theme().muted
    }

    // === SEMANTIC COLORS ===

    /// Success/positive action color
    fn success(&self) -> Hsla {
        self.theme().success
    }

    /// Error/danger color
    fn error(&self) -> Hsla {
        self.theme().danger
    }

    /// Warning color
    fn warning(&self) -> Hsla {
        self.theme().warning
    }

    /// Info/primary action color
    fn info(&self) -> Hsla {
        self.theme().info
    }
}

impl OneDarkExt for Theme {
    fn theme(&self) -> &Theme {
        self
    }
}

/// Typography constants based on Zed's text sizing system
/// Base rem size: 16px
pub struct Typography;

impl Typography {
    // === UI TEXT SIZES ===

    /// Large UI text: 16px (1.0rem)
    pub const TEXT_UI_LG: f32 = 16.0;

    /// Default UI text: 14px (0.875rem)
    pub const TEXT_UI: f32 = 14.0;

    /// Small UI text: 12px (0.75rem)
    pub const TEXT_UI_SM: f32 = 12.0;

    /// Extra small UI text: 10px (0.625rem)
    pub const TEXT_UI_XS: f32 = 10.0;

    // === HEADLINE SIZES ===

    /// Extra small headline: 14px
    pub const HEADLINE_XS: f32 = 14.0;

    /// Small headline: 16px
    pub const HEADLINE_SM: f32 = 16.0;

    /// Medium headline: 18px
    pub const HEADLINE_MD: f32 = 18.0;

    /// Large headline: 20px
    pub const HEADLINE_LG: f32 = 20.0;

    /// Extra large headline: 23px
    pub const HEADLINE_XL: f32 = 23.0;

    // === LINE HEIGHTS ===

    /// Standard line height for headlines
    pub const LINE_HEIGHT_HEADLINE: f32 = 25.6;

    /// Tight line height for UI text
    pub const LINE_HEIGHT_TIGHT: f32 = 1.2;

    /// Normal line height
    pub const LINE_HEIGHT_NORMAL: f32 = 1.5;

    /// Relaxed line height
    pub const LINE_HEIGHT_RELAXED: f32 = 1.6;
}

/// Spacing constants based on Zed's DynamicSpacing system
/// Values are for "Default" density (Compact and Comfortable have different values)
pub struct Spacing;

impl Spacing {
    /// 0px - No spacing
    pub const BASE_00: f32 = 0.0;

    /// 1px - Minimal spacing
    pub const BASE_01: f32 = 1.0;

    /// 2px - Extra small spacing
    pub const BASE_02: f32 = 2.0;

    /// 3px - Tiny spacing
    pub const BASE_03: f32 = 3.0;

    /// 4px - Small spacing
    pub const BASE_04: f32 = 4.0;

    /// 6px - Medium-small spacing
    pub const BASE_06: f32 = 6.0;

    /// 8px - Medium spacing
    pub const BASE_08: f32 = 8.0;

    /// 12px - Large spacing
    pub const BASE_12: f32 = 12.0;

    /// 16px - Extra large spacing
    pub const BASE_16: f32 = 16.0;

    /// 20px - Double extra large spacing
    pub const BASE_20: f32 = 20.0;

    /// 24px - Triple extra large spacing
    pub const BASE_24: f32 = 24.0;

    /// 32px - Huge spacing
    pub const BASE_32: f32 = 32.0;

    /// 40px - Extra huge spacing
    pub const BASE_40: f32 = 40.0;

    /// 48px - Massive spacing
    pub const BASE_48: f32 = 48.0;
}

/// Button sizing constants
pub struct ButtonSize;

impl ButtonSize {
    /// Large button height: 32px
    pub const LARGE_HEIGHT: f32 = 32.0;
    pub const LARGE_PADDING: f32 = Spacing::BASE_08;

    /// Medium button height: 28px
    pub const MEDIUM_HEIGHT: f32 = 28.0;
    pub const MEDIUM_PADDING: f32 = Spacing::BASE_08;

    /// Default button height: 22px
    pub const DEFAULT_HEIGHT: f32 = 22.0;
    pub const DEFAULT_PADDING: f32 = Spacing::BASE_04;

    /// Compact button height: 18px
    pub const COMPACT_HEIGHT: f32 = 18.0;
    pub const COMPACT_PADDING: f32 = Spacing::BASE_04;

    /// Minimal button height: 16px
    pub const MINIMAL_HEIGHT: f32 = 16.0;
    pub const MINIMAL_PADDING: f32 = Spacing::BASE_00;
}

/// Checkbox/toggle sizing constants
pub struct CheckboxSize;

impl CheckboxSize {
    /// Container size: 20x20px
    pub const CONTAINER: f32 = 20.0;

    /// Inner box size: 4x4px
    pub const INNER_BOX: f32 = 4.0;

    /// Border width: 1px
    pub const BORDER_WIDTH: f32 = 1.0;

    /// Border radius (small)
    pub const BORDER_RADIUS: f32 = 2.0;
}

/// Input field sizing constants
pub struct InputSize;

impl InputSize {
    /// Default input height: 30px (content) + 8px (padding) = 38px total
    pub const HEIGHT: f32 = 30.0;

    /// Input padding: 4px
    pub const PADDING: f32 = Spacing::BASE_04;

    /// Input text size: 14px (matches TEXT_UI)
    pub const TEXT_SIZE: f32 = Typography::TEXT_UI;

    /// Input line height
    pub const LINE_HEIGHT: f32 = 30.0;

    /// Cursor width: 2px
    pub const CURSOR_WIDTH: f32 = 2.0;
}

/// Border radius constants
pub struct BorderRadius;

impl BorderRadius {
    /// Extra small: 2px
    pub const XS: f32 = 2.0;

    /// Small: 4px
    pub const SM: f32 = 4.0;

    /// Medium: 4px (confirmed from Zed)
    pub const MD: f32 = 4.0;

    /// Large: 8px
    pub const LG: f32 = 8.0;

    /// Extra large: 12px
    pub const XL: f32 = 12.0;

    /// Window decoration rounding: 10px
    pub const WINDOW: f32 = 10.0;
}
