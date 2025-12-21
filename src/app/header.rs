//! Header rendering: app title bar and Windows-specific controls

use super::{colors, AppState};
use crate::theme::ThemeRegistry;
use crate::SwitchTheme;
use gpui::{
    Context, Corner, FontWeight, InteractiveElement, IntoElement, ParentElement, SharedString,
    Styled, WindowControlArea, prelude::FluentBuilder, px, svg,
};
use gpui_component::{
    Theme,
    button::{Button, ButtonVariants},
    h_flex,
    menu::DropdownMenu,
};

impl AppState {
    /// Render the header with app name (34px height)
    /// On macOS: 80px left padding for traffic lights
    /// On Windows: hamburger menu + minimize/close buttons, draggable area
    pub(crate) fn render_header(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let is_windows = cfg!(target_os = "windows");

        // Render hamburger menu first (Windows only) to avoid borrow issues
        let hamburger_menu = if is_windows {
            Some(self.render_hamburger_menu(cx).into_any_element())
        } else {
            None
        };

        let theme = Theme::global(cx);

        h_flex()
            .w_full()
            .h(px(34.0))
            .flex_shrink_0()
            .border_b_1()
            .border_color(theme.border)
            // Hamburger menu button (Windows only)
            .when_some(hamburger_menu, |this, menu| this.child(menu))
            // Draggable title area (contains label and fills remaining space)
            .child(
                h_flex()
                    .id("titlebar-drag-area")
                    .flex_1()
                    .h_full()
                    .items_center()
                    // On macOS: left padding for traffic lights. On Windows: no extra padding (hamburger is there)
                    .when(!is_windows, |this| this.pl(px(80.0)))
                    // Mark this area as a window drag region for the platform
                    .window_control_area(WindowControlArea::Drag)
                    // App name label
                    .child(
                        gpui::div()
                            .text_sm()
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(theme.foreground)
                            .child("aresampler"),
                    )
                    // Version number
                    .child(
                        gpui::div()
                            .text_sm()
                            .text_color(theme.muted_foreground)
                            .child(concat!(" v", env!("CARGO_PKG_VERSION"))),
                    ),
            )
            // Window control buttons (Windows only)
            .when(is_windows, |this| {
                this.child(
                    h_flex()
                        .items_center()
                        // Minimize button
                        .child(
                            gpui::div()
                                .id("minimize-button")
                                .w(px(46.0))
                                .h(px(34.0))
                                .flex()
                                .items_center()
                                .justify_center()
                                .cursor_pointer()
                                .hover(|this| this.bg(colors::bg_tertiary()))
                                .on_mouse_down(
                                    gpui::MouseButton::Left,
                                    cx.listener(|_this, _, window, _cx| {
                                        window.minimize_window();
                                    }),
                                )
                                .child(
                                    // Minimize icon (horizontal line)
                                    gpui::div().w(px(10.0)).h(px(1.0)).bg(colors::text_secondary()),
                                ),
                        )
                        // Close button
                        .child(
                            gpui::div()
                                .id("close-button")
                                .w(px(46.0))
                                .h(px(34.0))
                                .flex()
                                .items_center()
                                .justify_center()
                                .cursor_pointer()
                                .hover(|this| this.bg(colors::recording()))
                                .on_mouse_down(
                                    gpui::MouseButton::Left,
                                    cx.listener(|_this, _, window, _cx| {
                                        window.remove_window();
                                    }),
                                )
                                .child(
                                    // Close icon (X character)
                                    gpui::div()
                                        .text_xs()
                                        .text_color(colors::text_secondary())
                                        .child("✕"),
                                ),
                        ),
                )
            })
    }

    /// Render the theme palette button with dropdown theme picker (Windows only)
    pub(crate) fn render_hamburger_menu(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = Theme::global(cx);
        let current_theme_name = theme.theme_name().to_string();
        let icon_color = theme.muted_foreground;

        // Build the palette button with a dropdown menu showing themes directly
        Button::new("theme-menu")
            .ghost()
            .compact()
            .mr(px(8.0))
            .child(
                // Palette icon SVG
                svg()
                    .path("icons/palette.svg")
                    .size(px(16.0))
                    .text_color(icon_color),
            )
            .dropdown_menu_with_anchor(Corner::TopLeft, move |menu, _window, _cx| {
                // Build the theme list directly (scrollable)
                let current_name = current_theme_name.clone();
                let registry = ThemeRegistry::new();

                let mut menu = menu.scrollable(true).max_h(px(300.0));
                for theme_variant in &registry.themes {
                    let is_current = theme_variant.name == current_name;
                    let theme_name: SharedString = theme_variant.name.clone().into();
                    menu = menu.menu_with_check(
                        theme_variant.name.clone(),
                        is_current,
                        Box::new(SwitchTheme(theme_name)),
                    );
                }
                menu
            })
    }
}
