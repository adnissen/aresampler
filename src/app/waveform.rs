//! Waveform section rendering with trim controls

use super::AppState;
use crate::waveform::{WaveformData, WaveformView};
use gpui::{
    Bounds, Context, CursorStyle, InteractiveElement, IntoElement, MouseDownEvent, MouseMoveEvent,
    ParentElement, Size, StatefulInteractiveElement, Styled, prelude::FluentBuilder, point, px,
};
use gpui_component::{Theme, h_flex, v_flex};
use std::sync::Arc;

impl AppState {
    /// Render the waveform section with time display and controls
    pub(crate) fn render_waveform_section(
        &self,
        data: Arc<WaveformData>,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = Theme::global(cx);
        let trim_selection = self.trim_selection.clone();
        let is_modified = self.trim_selection.is_modified();
        let duration = data.duration_secs;

        // Calculate times based on trim selection
        let start_time = format!(
            "{}:{:02}",
            (duration * self.trim_selection.start as f64) as u32 / 60,
            (duration * self.trim_selection.start as f64) as u32 % 60
        );
        let current_time = format!(
            "{}:{:02}",
            (duration * 0.75) as u32 / 60,
            (duration * 0.75) as u32 % 60
        );
        let end_time = format!(
            "{}:{:02}",
            (duration * self.trim_selection.end as f64) as u32 / 60,
            (duration * self.trim_selection.end as f64) as u32 % 60
        );

        v_flex()
            .px_4()
            .py_3()
            .gap_3()
            // Waveform container
            .child(
                gpui::div()
                    .id("waveform-container")
                    .h(px(80.0))
                    .w_full()
                    .rounded(px(10.0))
                    .overflow_hidden()
                    .cursor(CursorStyle::ResizeLeftRight)
                    .on_mouse_down(
                        gpui::MouseButton::Left,
                        cx.listener(move |this, event: &MouseDownEvent, window, cx| {
                            let viewport = window.viewport_size();
                            let padding: f32 = 16.0;
                            let viewport_width: f32 = viewport.width.into();
                            let waveform_width = viewport_width - (padding * 2.0);
                            let bounds = Bounds {
                                origin: point(px(padding), event.position.y - px(40.0)),
                                size: Size {
                                    width: px(waveform_width),
                                    height: px(80.0),
                                },
                            };
                            this.waveform_bounds = Some(bounds);
                            this.handle_waveform_mouse_down(event.position, cx);
                        }),
                    )
                    .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _window, cx| {
                        this.handle_waveform_mouse_move(event.position, cx);
                    }))
                    .on_mouse_up(
                        gpui::MouseButton::Left,
                        cx.listener(|this, _, _window, cx| {
                            this.handle_waveform_mouse_up(cx);
                        }),
                    )
                    .on_hover(cx.listener(|this, hovered: &bool, _window, cx| {
                        if !*hovered {
                            this.handle_waveform_mouse_up(cx);
                        }
                    }))
                    .child(
                        WaveformView::new(data)
                            .with_trim_selection(trim_selection)
                            .with_background(theme.muted)
                            .with_color(theme.primary)
                            .with_handle_color(theme.primary)
                            .with_dimmed_color(gpui::hsla(0.0, 0.0, 0.0, 0.5))
                            .render(),
                    ),
            )
            // Time display row
            .child(
                h_flex()
                    .justify_between()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(start_time)
                    .child(gpui::div().text_color(theme.primary).child(current_time))
                    .child(end_time),
            )
            // Control buttons
            .child(
                h_flex()
                    .gap_2()
                    // Play button (icon only)
                    .child(
                        gpui::div()
                            .id("play-button")
                            .size(px(44.0))
                            .rounded(px(10.0))
                            .bg(theme.secondary)
                            .border_1()
                            .border_color(theme.border)
                            .flex()
                            .items_center()
                            .justify_center()
                            .cursor_pointer()
                            .hover(|s| s.bg(theme.secondary_hover))
                            .on_mouse_down(
                                gpui::MouseButton::Left,
                                cx.listener(|this, _, window, cx| {
                                    this.toggle_playback(window, cx);
                                }),
                            )
                            .child(
                                gpui::div()
                                    .text_color(theme.foreground)
                                    .child(if self.is_playing { "⏹" } else { "▶" }),
                            ),
                    )
                    // Cut Selection button
                    .child(
                        gpui::div()
                            .id("cut-button")
                            .flex_1()
                            .py_3()
                            .rounded(px(10.0))
                            .bg(theme.secondary)
                            .border_1()
                            .border_color(theme.border)
                            .flex()
                            .items_center()
                            .justify_center()
                            .cursor_pointer()
                            .text_sm()
                            .text_color(theme.foreground)
                            .when(!is_modified, |this| this.opacity(0.5).cursor_not_allowed())
                            .when(is_modified, |this| {
                                this.hover(|s| s.bg(theme.secondary_hover)).on_mouse_down(
                                    gpui::MouseButton::Left,
                                    cx.listener(|this, _, window, cx| {
                                        this.cut_audio(window, cx);
                                    }),
                                )
                            })
                            .child("Cut Selection"),
                    )
                    // Reset button (icon only, same size as play button)
                    .child(
                        gpui::div()
                            .id("reset-button")
                            .size(px(44.0))
                            .rounded(px(10.0))
                            .bg(theme.secondary)
                            .border_1()
                            .border_color(theme.border)
                            .flex()
                            .items_center()
                            .justify_center()
                            .cursor_pointer()
                            .hover(|s| s.bg(theme.secondary_hover))
                            .on_mouse_down(
                                gpui::MouseButton::Left,
                                cx.listener(|this, _, window, cx| {
                                    this.reset_session(window, cx);
                                }),
                            )
                            .child(gpui::div().text_color(theme.foreground).child("↺")),
                    ),
            )
    }
}
