//! Output file card rendering and stats row

use super::AppState;
use gpui::{
    Context, FontWeight, InteractiveElement, IntoElement, ParentElement,
    StatefulInteractiveElement, Styled, prelude::FluentBuilder, px,
};
use gpui_component::{
    Theme,
    h_flex, v_flex,
};

impl AppState {
    /// Render the output file selection card
    pub(crate) fn render_output_card(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = Theme::global(cx);
        let has_output = self.output_path.is_some();
        let has_waveform = self.waveform_data.is_some();
        let is_locked = has_waveform || self.is_recording;
        let output_name = self
            .output_path
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "Choose destination...".to_string());

        // Capture theme colors for closures
        let theme_border = theme.border;
        let theme_muted = theme.muted;
        let theme_muted_fg = theme.muted_foreground;

        // Label text changes based on state
        let label_text = if has_waveform { "SAVED AS" } else { "SAVE TO" };

        gpui::div()
            .px_4()
            .py_3()
            .border_b_1()
            .border_color(theme_border)
            .child(
                gpui::div()
                    .id("output-card")
                    .w_full()
                    .px_3()
                    .py_2()
                    .rounded(px(10.0))
                    // Grey out and disable interaction when locked (recording or waveform displayed)
                    .when(is_locked, |this| this.opacity(0.5))
                    .when(!is_locked, |this| {
                        this.cursor_pointer().on_mouse_down(
                            gpui::MouseButton::Left,
                            cx.listener(|this, _, window, cx| {
                                this.browse_output(window, cx);
                            }),
                        )
                    })
                    .when(has_output, |this| this.bg(theme_muted))
                    .when(!has_output, |this| {
                        this.border_1().border_color(theme_border)
                    })
                    .child(
                        h_flex()
                            .gap_3()
                            .items_center()
                            // File icon
                            .child(self.render_file_icon(has_output, cx))
                            // Text content
                            .child(
                                v_flex()
                                    .flex_1()
                                    .gap(px(2.0))
                                    .child(
                                        gpui::div()
                                            .text_xs()
                                            .text_color(theme_muted_fg)
                                            .child(label_text),
                                    )
                                    .child(
                                        gpui::div()
                                            .text_sm()
                                            .font_weight(FontWeight::MEDIUM)
                                            .when(!has_output, |this| {
                                                this.text_color(theme_muted_fg)
                                            })
                                            .child(output_name),
                                    ),
                            ),
                    ),
            )
    }

    /// Render the file icon (accent color when selected, muted when empty)
    pub(crate) fn render_file_icon(&self, has_output: bool, cx: &Context<Self>) -> impl IntoElement {
        let theme = Theme::global(cx);

        gpui::div()
            .size(px(28.0))
            .rounded(px(6.0))
            .flex()
            .items_center()
            .justify_center()
            .when(has_output, |this| this.bg(theme.primary))
            .when(!has_output, |this| this.bg(theme.muted))
            .child(
                gpui::div()
                    .text_xs()
                    .text_color(if has_output {
                        theme.primary_foreground
                    } else {
                        theme.muted_foreground
                    })
                    .child("📄"),
            )
    }

    /// Render the horizontal stats row
    pub(crate) fn render_stats_row(&self, cx: &Context<Self>) -> impl IntoElement {
        let theme = Theme::global(cx);

        // Total duration includes any captured pre-roll
        let total_duration = self.stats.duration_secs + self.captured_pre_roll_secs as f64;
        let duration = format!("{:.1}s", total_duration);
        let size = format!(
            "{:.1} MB",
            self.stats.file_size_bytes as f64 / (1024.0 * 1024.0)
        );

        h_flex()
            .px_4()
            .py_3()
            .gap_3()
            .justify_center()
            .border_b_1()
            .border_color(theme.border)
            // Duration stat
            .child(
                v_flex()
                    .w(px(120.0))
                    .items_center()
                    .p_3()
                    .bg(theme.muted)
                    .rounded_md()
                    .child(
                        gpui::div()
                            .text_base()
                            .font_weight(FontWeight::MEDIUM)
                            .child(duration),
                    )
                    .child(
                        gpui::div()
                            .text_xs()
                            .text_color(theme.muted_foreground)
                            .child("Duration"),
                    ),
            )
            // Size stat
            .child(
                v_flex()
                    .w(px(120.0))
                    .items_center()
                    .p_3()
                    .bg(theme.muted)
                    .rounded_md()
                    .child(
                        gpui::div()
                            .text_base()
                            .font_weight(FontWeight::MEDIUM)
                            .child(size),
                    )
                    .child(
                        gpui::div()
                            .text_xs()
                            .text_color(theme.muted_foreground)
                            .child("Size"),
                    ),
            )
    }
}
