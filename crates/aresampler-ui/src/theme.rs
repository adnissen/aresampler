//! Theme integration with gpui-component
//!
//! This module loads themes and provides helper functions
//! for accessing theme colors throughout the application.

use gpui::{App, Hsla};
use gpui_component::{Theme, ThemeConfig, ThemeSet};
use std::rc::Rc;

/// Embedded theme files
const ADVENTURE_THEME: &str = include_str!("../assets/themes/adventure.json");
const ALDUIN_THEME: &str = include_str!("../assets/themes/alduin.json");
const AYU_THEME: &str = include_str!("../assets/themes/ayu.json");
const CATPPUCCIN_THEME: &str = include_str!("../assets/themes/catppuccin.json");
const EVERFOREST_THEME: &str = include_str!("../assets/themes/everforest.json");
const FAHRENHEIT_THEME: &str = include_str!("../assets/themes/fahrenheit.json");
const FLEXOKI_THEME: &str = include_str!("../assets/themes/flexoki.json");
const GRUVBOX_THEME: &str = include_str!("../assets/themes/gruvbox.json");
const HARPER_THEME: &str = include_str!("../assets/themes/harper.json");
const HYBRID_THEME: &str = include_str!("../assets/themes/hybrid.json");
const JELLYBEANS_THEME: &str = include_str!("../assets/themes/jellybeans.json");
const KIBBLE_THEME: &str = include_str!("../assets/themes/kibble.json");
const MACOS_CLASSIC_THEME: &str = include_str!("../assets/themes/macos-classic.json");
const MATRIX_THEME: &str = include_str!("../assets/themes/matrix.json");
const MELLIFLUOUS_THEME: &str = include_str!("../assets/themes/mellifluous.json");
const MOLOKAI_THEME: &str = include_str!("../assets/themes/molokai.json");
const ONE_DARK_THEME: &str = include_str!("../assets/themes/one-dark-theme.json");
const SOLARIZED_THEME: &str = include_str!("../assets/themes/solarized.json");
const SPACEDUCK_THEME: &str = include_str!("../assets/themes/spaceduck.json");
const TOKYONIGHT_THEME: &str = include_str!("../assets/themes/tokyonight.json");
const TWILIGHT_THEME: &str = include_str!("../assets/themes/twilight.json");

/// A single theme variant that can be applied
#[derive(Clone)]
pub struct ThemeVariant {
    /// Display name for the menu
    pub name: String,
    /// The theme configuration
    pub config: ThemeConfig,
}

/// Registry of all available themes
pub struct ThemeRegistry {
    /// All available theme variants
    pub themes: Vec<ThemeVariant>,
}

impl ThemeRegistry {
    /// Create the theme registry with all embedded themes
    pub fn new() -> Self {
        let mut themes = Vec::new();

        // Helper to parse and add themes from a JSON string
        let mut add_themes = |json: &str| {
            if let Ok(theme_set) = serde_json::from_str::<ThemeSet>(json) {
                for config in theme_set.themes {
                    themes.push(ThemeVariant {
                        name: config.name.to_string(),
                        config,
                    });
                }
            }
        };

        // Add all themes - sorted alphabetically by file name
        add_themes(ADVENTURE_THEME);
        add_themes(ALDUIN_THEME);
        add_themes(AYU_THEME);
        add_themes(CATPPUCCIN_THEME);
        add_themes(EVERFOREST_THEME);
        add_themes(FAHRENHEIT_THEME);
        add_themes(FLEXOKI_THEME);
        add_themes(GRUVBOX_THEME);
        add_themes(HARPER_THEME);
        add_themes(HYBRID_THEME);
        add_themes(JELLYBEANS_THEME);
        add_themes(KIBBLE_THEME);
        add_themes(MACOS_CLASSIC_THEME);
        add_themes(MATRIX_THEME);
        add_themes(MELLIFLUOUS_THEME);
        add_themes(MOLOKAI_THEME);
        add_themes(ONE_DARK_THEME);
        add_themes(SOLARIZED_THEME);
        add_themes(SPACEDUCK_THEME);
        add_themes(TOKYONIGHT_THEME);
        add_themes(TWILIGHT_THEME);

        // Sort themes by name for consistent menu ordering
        themes.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));

        Self { themes }
    }

    /// Get all theme names for menu display
    pub fn theme_names(&self) -> Vec<String> {
        self.themes.iter().map(|t| t.name.clone()).collect()
    }

    /// Find a theme by name
    pub fn find_theme(&self, name: &str) -> Option<&ThemeVariant> {
        self.themes.iter().find(|t| t.name == name)
    }
}

impl Default for ThemeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply a theme configuration to the global theme
pub fn apply_theme(config: &ThemeConfig, cx: &mut App) {
    let theme = Theme::global_mut(cx);
    let config_rc = Rc::new(config.clone());

    // Set as appropriate theme based on mode
    if config.mode.is_dark() {
        theme.dark_theme = config_rc.clone();
    } else {
        theme.light_theme = config_rc.clone();
    }

    // Apply the configuration
    theme.apply_config(&config_rc);

    // Disable shadows globally (helps with context menu shadow artifacts)
    theme.shadow = false;
}

/// Initialize the default theme (One Dark) for the application.
/// Call this after `gpui_component::init(cx)` in your main function.
pub fn init(cx: &mut App) {
    init_with_theme_name(None, cx);
}

/// Initialize the application with a specific theme by name.
/// Falls back to "One Dark" if the theme is not found.
pub fn init_with_theme_name(theme_name: Option<&str>, cx: &mut App) {
    let registry = ThemeRegistry::new();

    // Try to find the saved theme, fall back to One Dark
    let config = theme_name
        .and_then(|name| registry.find_theme(name))
        .or_else(|| registry.find_theme("One Dark"))
        .map(|v| v.config.clone());

    if let Some(config) = config {
        apply_theme(&config, cx);
    }
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
