//! Application state and main UI rendering

// Submodules
mod colors;
mod header;
mod lifecycle;
mod output;
mod playback;
mod recording;
mod sources;
mod waveform;

// Re-export types for use elsewhere in the crate
pub use playback::AudioPlayer;
pub use waveform::{DragHandle, TrimSelection, WaveformData, WaveformError};

use crate::source_selection::SourceSelectionState;
use crate::core::{CaptureSession, CaptureStats, CaptureEvent};
use gpui::{
    Bounds, Context, ElementId, FontWeight, InteractiveElement, IntoElement, ParentElement, Pixels,
    Render, Styled, Window, prelude::FluentBuilder, px, relative, rgb,
};
use gpui_component::{
    Theme,
    button::{Button, ButtonVariants},
    h_flex,
    scroll::ScrollableElement,
    v_flex,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::Instant;

/// Main application state
pub struct AppState {
    // Permission state
    pub(crate) has_permission: bool,
    pub(crate) permission_error: Option<String>,

    // Source selection (process/application picker)
    pub(crate) source_selection: SourceSelectionState,

    // Output file
    pub(crate) output_path: Option<PathBuf>,

    // Pre-roll / Monitoring state
    pub(crate) pre_roll_seconds: f32,
    pub(crate) sample_rate: u32,
    pub(crate) is_monitoring: bool,

    // Recording state
    pub(crate) is_recording: bool,
    pub(crate) stats: CaptureStats,
    pub(crate) captured_pre_roll_secs: f32, // Pre-roll duration captured when recording started
    pub(crate) capture_session: Option<CaptureSession>,
    pub(crate) event_receiver: Option<Receiver<CaptureEvent>>,

    // Error message
    pub(crate) error_message: Option<String>,

    // Last time we triggered a render update
    pub(crate) last_render_time: Option<Instant>,

    // Waveform data loaded after recording completes
    pub(crate) waveform_data: Option<Arc<WaveformData>>,
    pub(crate) waveform_loading: bool,

    // Trim selection state
    pub(crate) trim_selection: TrimSelection,
    pub(crate) dragging_handle: Option<DragHandle>,
    pub(crate) waveform_bounds: Option<Bounds<Pixels>>,

    // Audio playback state
    pub(crate) audio_player: Option<AudioPlayer>,
    pub(crate) is_playing: bool,
}

impl Render for AppState {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Poll for capture events during render (simple approach)
        if self.is_recording || self.is_monitoring {
            self.poll_capture_events();
            // Request another frame to continue polling, but only notify every 100ms
            Self::schedule_next_frame_check(window, cx);
        }

        // If we don't have permission, show permission request UI
        if !self.has_permission {
            return v_flex()
                .size_full()
                .bg(colors::bg_primary())
                .text_color(colors::text_primary())
                .child(self.render_header(cx))
                .child(
                    v_flex()
                        .flex_1()
                        .p_4()
                        .gap_4()
                        .child(
                            v_flex()
                                .gap_2()
                                .child(
                                    gpui::div()
                                        .text_lg()
                                        .font_weight(FontWeight::BOLD)
                                        .child("Permission Required"),
                                )
                                .child(
                                    gpui::div()
                                        .text_sm()
                                        .text_color(colors::text_secondary())
                                        .child("Screen Recording permission is required to capture audio from applications."),
                                ),
                        )
                        .child(
                            Button::new("request_permission")
                                .label("Request Permission")
                                .primary()
                                .on_click(cx.listener(|this, _, window, cx| {
                                    this.request_permission(window, cx);
                                })),
                        )
                        .when_some(self.permission_error.clone(), |this, msg| {
                            this.child(
                                gpui::div()
                                    .p_2()
                                    .bg(colors::error_bg())
                                    .rounded_md()
                                    .text_sm()
                                    .text_color(colors::error_text())
                                    .child(msg),
                            )
                        }),
                )
                .into_any_element();
        }

        let can_record = self.source_selection.has_any_selection()
            && self.output_path.is_some()
            && !self.is_recording;

        let record_button_label = if self.is_recording {
            "Stop Recording"
        } else {
            "Start Recording"
        };

        // Pre-roll values for the segmented toggle
        let pre_roll_options: [(f32, &str); 4] =
            [(0.0, "Off"), (5.0, "5s"), (10.0, "10s"), (30.0, "30s")];

        let theme = Theme::global(cx);

        // Capture theme colors for use in closures
        let theme_border = theme.border;
        let theme_muted = theme.muted;
        let theme_muted_fg = theme.muted_foreground;
        let _theme_secondary = theme.secondary;
        let theme_secondary_active = theme.secondary_active;
        let theme_primary = theme.primary;

        v_flex()
            .size_full()
            .bg(theme.background)
            .text_color(theme.foreground)
            // Header with logo
            .child(self.render_header(cx))
            // Main content
            .child(
                v_flex()
                    .flex_1()
                    .pb_8()
                    .gap_0()
                    // Source Selection Cards
                    .child(self.render_sources_section(cx))
                    // Arrow divider between source and output
                    .child(self.render_arrow_divider(cx))
                    // Output File Card
                    .child(self.render_output_card(cx))
                    // Pre-roll Toggle Row (hidden after recording complete)
                    .when(self.waveform_data.is_none() || self.is_recording, |this| {
                        this.child(
                            h_flex()
                                .px_4()
                                .py_3()
                                .items_center()
                                .justify_between()
                                .border_b_1()
                                .border_color(theme_border)
                                .child(
                                    gpui::div()
                                        .w(relative(0.5))
                                        .relative()
                                        .child(
                                            gpui::div()
                                                .top_neg_3()
                                                .text_xs()
                                                .text_color(theme_muted_fg)
                                                .child("Pre-roll buffer"),
                                        )
                                        .child(
                                            v_flex()
                                                .absolute()
                                                .top_1p5()
                                                .left_0()
                                                .w(relative(0.8))
                                                .text_color(theme_muted_fg)
                                                .text_size(px(9.0))
                                                .line_height(px(10.0))
                                                .child("time saved before recording starts"),
                                        ),
                                )
                                .child(
                                    h_flex()
                                        .gap_1()
                                        .p_1()
                                        .bg(theme_muted)
                                        .rounded_md()
                                        .children(pre_roll_options.iter().enumerate().map(
                                            |(idx, (value, label))| {
                                                let is_active =
                                                    (self.pre_roll_seconds - *value).abs() < 0.1;
                                                let value = *value;
                                                let is_disabled = self.is_recording;

                                                gpui::div()
                                                    .id(ElementId::Name(
                                                        format!("preroll_{}", idx).into(),
                                                    ))
                                                    .px_2()
                                                    .py_1()
                                                    .rounded(px(6.0))
                                                    .text_xs()
                                                    .cursor_pointer()
                                                    .when(is_active, |this| {
                                                        this.bg(theme_secondary_active)
                                                            .text_color(theme_primary)
                                                    })
                                                    .when(!is_active, |this| {
                                                        this.text_color(theme_muted_fg)
                                                    })
                                                    .when(is_disabled, |this| {
                                                        this.opacity(0.5).cursor_not_allowed()
                                                    })
                                                    .when(!is_disabled, |this| {
                                                        this.on_mouse_down(
                                                            gpui::MouseButton::Left,
                                                            cx.listener(
                                                                move |this, _, _window, cx| {
                                                                    if this.is_monitoring {
                                                                        // Already monitoring - resize the buffer (works for 0 too)
                                                                        if let Some(session) =
                                                                            &mut this
                                                                                .capture_session
                                                                        {
                                                                            let _ = session
                                                                                .resize_pre_roll(
                                                                                    value,
                                                                                );
                                                                        }
                                                                    } else if this.source_selection.has_any_selection() {
                                                                        // Not monitoring but has sources - start monitoring
                                                                        this.start_monitoring(cx);
                                                                    }
                                                                    this.pre_roll_seconds = value;
                                                                    cx.notify();
                                                                },
                                                            ),
                                                        )
                                                    })
                                                    .child(*label)
                                            },
                                        )),
                                ),
                        )
                    })
                    // Sample Rate Toggle Row (hidden after recording complete)
                    .when(self.waveform_data.is_none() || self.is_recording, |this| {
                        // ScreenCaptureKit supported sample rates: 8000, 16000, 24000, 48000 Hz
                        // https://developer.apple.com/documentation/screencapturekit/scstreamconfiguration/samplerate
                        let sample_rate_options: [(u32, &str); 4] =
                            [(8000, "8k"), (16000, "16k"), (24000, "24k"), (48000, "48k")];
                        let current_sample_rate = self.sample_rate;
                        let is_disabled = self.is_recording;

                        this.child(
                            h_flex()
                                .px_4()
                                .py_3()
                                .items_center()
                                .justify_between()
                                .border_b_1()
                                .border_color(theme_border)
                                .child(
                                    gpui::div()
                                        .w(relative(0.5))
                                        .relative()
                                        .child(
                                            gpui::div()
                                                .top_neg_3()
                                                .text_xs()
                                                .text_color(theme_muted_fg)
                                                .child("Sample rate"),
                                        )
                                        .child(
                                            v_flex()
                                                .absolute()
                                                .top_1p5()
                                                .left_0()
                                                .w(relative(0.8))
                                                .text_color(theme_muted_fg)
                                                .text_size(px(9.0))
                                                .line_height(px(10.0))
                                                .child("output audio quality"),
                                        ),
                                )
                                .child(
                                    h_flex()
                                        .gap_1()
                                        .p_1()
                                        .bg(theme_muted)
                                        .rounded_md()
                                        .children(sample_rate_options.iter().enumerate().map(
                                            |(idx, (value, label))| {
                                                let is_active = current_sample_rate == *value;
                                                let value = *value;

                                                gpui::div()
                                                    .id(ElementId::Name(
                                                        format!("samplerate_{}", idx).into(),
                                                    ))
                                                    .px_2()
                                                    .py_1()
                                                    .rounded(px(6.0))
                                                    .text_xs()
                                                    .cursor_pointer()
                                                    .when(is_active, |this| {
                                                        this.bg(theme_secondary_active)
                                                            .text_color(theme_primary)
                                                    })
                                                    .when(!is_active, |this| {
                                                        this.text_color(theme_muted_fg)
                                                    })
                                                    .when(is_disabled, |this| {
                                                        this.opacity(0.5).cursor_not_allowed()
                                                    })
                                                    .when(!is_disabled, |this| {
                                                        this.on_mouse_down(
                                                            gpui::MouseButton::Left,
                                                            cx.listener(
                                                                move |this, _, _window, cx| {
                                                                    this.sample_rate = value;
                                                                    // Restart monitoring if active to apply new sample rate
                                                                    if this.is_monitoring {
                                                                        this.start_monitoring(cx);
                                                                    }
                                                                    cx.notify();
                                                                },
                                                            ),
                                                        )
                                                    })
                                                    .child(*label)
                                            },
                                        )),
                                ),
                        )
                    })
                    // Record Button (hidden after recording complete)
                    .when(self.waveform_data.is_none() || self.is_recording, |this| {
                        this.child(
                            gpui::div().px_4().py_3().child(
                                gpui::div()
                                    .id("record-button")
                                    .w_full()
                                    .py_3()
                                    .rounded(px(10.0))
                                    .text_sm()
                                    .font_weight(FontWeight::MEDIUM)
                                    .cursor_pointer()
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .gap_2()
                                    .when(self.is_recording, |this| {
                                        this.bg(colors::recording()).text_color(rgb(0xffffff))
                                    })
                                    .when(!self.is_recording && can_record, |this| {
                                        this.bg(colors::recording()).text_color(rgb(0xffffff))
                                    })
                                    .when(!self.is_recording && !can_record, |this| {
                                        this.bg(colors::recording())
                                            .text_color(rgb(0xffffff))
                                            .opacity(0.4)
                                            .cursor_not_allowed()
                                    })
                                    .when(can_record || self.is_recording, |this| {
                                        this.on_mouse_down(
                                            gpui::MouseButton::Left,
                                            cx.listener(|this, _, window, cx| {
                                                this.toggle_recording(window, cx);
                                            }),
                                        )
                                    })
                                    // Record icon (circle)
                                    .child(gpui::div().size(px(12.0)).rounded_full().bg(rgb(0xffffff)))
                                    .child(record_button_label),
                            ),
                        )
                    })
                    // Stats Row (shown during/after recording)
                    .when(self.is_recording || self.waveform_data.is_some(), |this| {
                        this.child(self.render_stats_row(cx))
                    })
                    // Waveform Section (shown after recording)
                    .when_some(self.waveform_data.clone(), |this, data| {
                        this.child(self.render_waveform_section(data, cx))
                    })
                    .when(self.waveform_loading, |this| {
                        this.child(
                            gpui::div()
                                .px_4()
                                .py_2()
                                .text_sm()
                                .text_color(colors::text_muted())
                                .child("Loading waveform..."),
                        )
                    })
                    // Error message
                    .when_some(self.error_message.clone(), |this, msg| {
                        this.child(
                            gpui::div()
                                .mx_4()
                                .my_2()
                                .p_2()
                                .bg(colors::error_bg())
                                .rounded_md()
                                .text_sm()
                                .text_color(colors::error_text())
                                .child(msg),
                        )
                    })
                    .overflow_y_scrollbar(),
            )
            .into_any_element()
    }
}

impl AppState {
    /// Render the arrow divider between source and output cards
    fn render_arrow_divider(&self, cx: &Context<Self>) -> impl IntoElement {
        let theme = Theme::global(cx);

        h_flex()
            .w_full()
            .px_4()
            .items_center()
            .gap_3()
            // Left line segment
            .child(gpui::div().flex_1().h(px(1.0)).bg(theme.border))
            // Arrow icon in the middle
            .child(
                gpui::div()
                    .text_sm()
                    .text_color(theme.muted_foreground)
                    .child("↓"),
            )
            // Right line segment
            .child(gpui::div().flex_1().h(px(1.0)).bg(theme.border))
    }
}
