//! Source management: adding, removing, subscribing to audio sources
//! Also contains source card rendering helpers

use super::{AppState, colors};
use crate::source_selection::{SourceEntry, SourceItem, render_placeholder_icon};
use gpui::{
    Context, ElementId, FontWeight, ImageSource, InteractiveElement, IntoElement, ParentElement,
    Styled, Window, img, prelude::FluentBuilder, px, relative,
};
use gpui_component::{
    Theme, h_flex,
    select::{SearchableVec, Select, SelectEvent},
    v_flex,
};

impl AppState {
    /// Subscribe to select events for a source entry.
    pub(crate) fn subscribe_to_source(source: &SourceEntry, index: usize, cx: &mut Context<Self>) {
        cx.subscribe(
            &source.select_state,
            move |this, _, event: &SelectEvent<SearchableVec<SourceItem>>, cx| {
                if let SelectEvent::Confirm(Some(value)) = event {
                    // Find the selected source by value
                    if let Some(source_item) = this.source_selection.find_source(value) {
                        this.source_selection.set_selected(index, source_item);
                        this.error_message = None;

                        // Update other dropdowns to filter out this selection
                        // Note: This is handled in refresh_source_dropdown when dropdown opens

                        // Start/restart monitoring when any source is selected
                        // (works for both apps and microphones)
                        if this.source_selection.has_any_selection() {
                            this.start_monitoring(cx);
                        }

                        cx.notify();
                    }
                }
            },
        )
        .detach();
    }

    /// Add a new source and subscribe to its events.
    pub(crate) fn add_source(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let index = self.source_selection.add_source(window, cx);
        let source = &self.source_selection.sources[index];
        Self::subscribe_to_source(source, index, cx);

        // Restart monitoring to include all current sources with aligned timing
        // Note: The new source isn't selected yet, but when it is, subscribe_to_source
        // will restart monitoring. This handles the case where user adds a source
        // while already monitoring - the new source will be included once selected.

        cx.notify();
    }

    /// Remove a source by index.
    pub(crate) fn remove_source(
        &mut self,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.source_selection.remove_source(index);
        // Update all dropdowns after removal
        self.source_selection.update_all_dropdowns(window, cx);

        // Restart monitoring with remaining sources to clear pre-roll buffer
        // and ensure timing alignment
        if self.is_monitoring {
            if self.source_selection.has_any_selection() {
                self.start_monitoring(cx);
            } else {
                // No sources left, stop monitoring
                self.stop_monitoring(cx);
            }
        }

        cx.notify();
    }

    /// Render all source selection cards plus the Add Source button.
    pub(crate) fn render_sources_section(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = Theme::global(cx);
        let source_count = self.source_selection.sources.len();

        // Capture theme colors for closures
        let theme_border = theme.border;
        let theme_muted_fg = theme.muted_foreground;
        let theme_muted = theme.muted;

        v_flex()
            .gap_0()
            // Render each source card with "+" dividers between them
            .children((0..source_count).flat_map(|index| {
                let mut elements: Vec<gpui::AnyElement> = Vec::new();
                // Add "+" divider before each card except the first
                if index > 0 {
                    elements.push(self.render_plus_divider(cx).into_any_element());
                }
                elements.push(self.render_source_card(index, cx).into_any_element());
                elements
            }))
            // Add Source button (only show if at least one source is selected, not recording, and no waveform)
            .when(
                self.source_selection.has_any_selection()
                    && !self.is_recording
                    && self.waveform_data.is_none(),
                |this| {
                    this.child(
                        gpui::div().px_4().pb_2().child(
                            gpui::div()
                                .id("add-source-button")
                                .w_full()
                                .py_2()
                                .rounded(px(8.0))
                                .border_1()
                                .border_color(theme_border)
                                .border_dashed()
                                .cursor_pointer()
                                .flex()
                                .items_center()
                                .justify_center()
                                .gap_2()
                                .text_sm()
                                .text_color(theme_muted_fg)
                                .hover(|this| this.bg(theme_muted))
                                .on_mouse_down(
                                    gpui::MouseButton::Left,
                                    cx.listener(|this, _, window, cx| {
                                        this.add_source(window, cx);
                                    }),
                                )
                                .child("+")
                                .child("Add Source"),
                        ),
                    )
                },
            )
    }

    /// Render the "+" divider between source cards
    pub(crate) fn render_plus_divider(&self, cx: &Context<Self>) -> impl IntoElement {
        let theme = Theme::global(cx);

        h_flex()
            .w_full()
            .px_4()
            .items_center()
            .gap_3()
            .h(px(5.0))
            // Left space segment
            .child(gpui::div().flex_1().h(px(1.0)))
            // Plus icon in the middle
            .child(
                gpui::div()
                    .text_sm()
                    .text_color(theme.muted_foreground)
                    .child("+"),
            )
            // Right space segment
            .child(gpui::div().flex_1().h(px(1.0)))
    }

    /// Render a single source (application or microphone) selection card
    pub(crate) fn render_source_card(
        &self,
        index: usize,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let theme = Theme::global(cx);
        let source = &self.source_selection.sources[index];
        let has_selection = source.selected_source.is_some();
        let is_first = index == 0;

        // Get level for this source (if available)
        let source_level = self.get_source_level(index);
        let is_active = self.stats.is_monitoring || self.stats.is_recording;
        let has_waveform = self.waveform_data.is_some();
        let is_locked = has_waveform || self.is_recording;

        // Capture theme colors for closures
        let theme_muted = theme.muted;
        let theme_muted_fg = theme.muted_foreground;
        let theme_border = theme.border;
        let theme_secondary = theme.secondary;
        let theme_danger = theme.danger;

        // Label text changes based on state and source type
        let label_text = if has_waveform {
            "RECORDED FROM"
        } else {
            "RECORDING FROM"
        };

        gpui::div().px_4().py_3().child(
            // Container with relative positioning for the overlay
            gpui::div()
                .relative()
                .w_full()
                // Grey out when recording or waveform is displayed
                .when(is_locked, |this| this.opacity(0.5))
                .child(
                    // Custom styled card (visual only)
                    gpui::div()
                        .w_full()
                        .px_3()
                        .py_2()
                        .rounded(px(10.0))
                        .when(has_selection, |this| this.bg(theme_muted))
                        .when(!has_selection, |this| {
                            this.border_1().border_color(theme_border)
                        })
                        .child(
                            v_flex()
                                .gap_2()
                                // Top row: icon, text, buttons
                                .child(
                                    h_flex()
                                        .gap_3()
                                        .items_center()
                                        // Icon (different for app vs microphone)
                                        .child(match &source.selected_source {
                                            Some(SourceItem::App(process)) => {
                                                if let Some(icon) = &process.icon {
                                                    gpui::div()
                                                        .size(px(28.0))
                                                        .rounded(px(6.0))
                                                        .overflow_hidden()
                                                        .child(
                                                            img(ImageSource::Render(icon.clone()))
                                                                .size(px(28.0)),
                                                        )
                                                        .into_any_element()
                                                } else {
                                                    render_placeholder_icon(cx).into_any_element()
                                                }
                                            }
                                            Some(SourceItem::Microphone(_)) => {
                                                // Microphone icon
                                                gpui::div()
                                                    .size(px(28.0))
                                                    .rounded(px(6.0))
                                                    .flex()
                                                    .items_center()
                                                    .justify_center()
                                                    .text_sm()
                                                    .child("🎤")
                                                    .into_any_element()
                                            }
                                            None => render_placeholder_icon(cx).into_any_element(),
                                        })
                                        // Text content
                                        .child(
                                            v_flex()
                                                .flex_1()
                                                .overflow_hidden()
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
                                                        .overflow_hidden()
                                                        .text_ellipsis()
                                                        .when(!has_selection, |this| {
                                                            this.text_color(theme_muted_fg)
                                                        })
                                                        .child(
                                                            source
                                                                .selected_source
                                                                .as_ref()
                                                                .map(|s| s.name().to_string())
                                                                .unwrap_or_else(|| {
                                                                    "Select source...".to_string()
                                                                }),
                                                        ),
                                                ),
                                        )
                                        // Remove button (X) for non-first sources (hide when locked)
                                        .when(!is_first && !is_locked, |this| {
                                            this.child(
                                                gpui::div()
                                                    .id(ElementId::Name(
                                                        format!("remove-source-{}", index).into(),
                                                    ))
                                                    .size(px(20.0))
                                                    .rounded(px(4.0))
                                                    .flex()
                                                    .items_center()
                                                    .justify_center()
                                                    .cursor_pointer()
                                                    .text_xs()
                                                    .text_color(theme_muted_fg)
                                                    .hover(|this| {
                                                        this.bg(theme_secondary)
                                                            .text_color(theme_danger)
                                                    })
                                                    .on_mouse_down(
                                                        gpui::MouseButton::Left,
                                                        cx.listener(move |this, _, window, cx| {
                                                            this.remove_source(index, window, cx);
                                                        }),
                                                    )
                                                    .child("✕"),
                                            )
                                        })
                                        // Chevron (only show for first source, hide when locked)
                                        .when(is_first && !is_locked, |this| {
                                            this.child(
                                                gpui::div().text_color(theme_muted_fg).child("▼"),
                                            )
                                        }),
                                )
                                // Level meter (inside the card, below the content row) - hide when waveform displayed (but show during recording)
                                .when(has_selection && is_active && !has_waveform, |this| {
                                    this.child(self.render_level_meter(source_level))
                                }),
                        ),
                )
                // Invisible Select overlay (handles click and dropdown) - disable when locked
                .when(!is_locked, |this| {
                    this.child(
                        gpui::div()
                            .id(ElementId::Name(
                                format!("source-select-overlay-{}", index).into(),
                            ))
                            .absolute()
                            .top_0()
                            .left_0()
                            .w_full()
                            .h_full()
                            .cursor_pointer()
                            .on_mouse_down(
                                gpui::MouseButton::Left,
                                cx.listener(move |this, _, window, cx| {
                                    // Refresh this source's dropdown before it opens
                                    this.source_selection
                                        .refresh_source_dropdown(index, window, cx);
                                }),
                            )
                            .child(
                                Select::new(&source.select_state)
                                    .w_full()
                                    .h_full()
                                    .appearance(false)
                                    .opacity(0.0)
                                    .placeholder("Select application..."),
                            ),
                    )
                }),
        )
    }

    /// Get the level for a specific source by index
    pub(crate) fn get_source_level(&self, index: usize) -> f32 {
        let source = &self.source_selection.sources[index];
        match &source.selected_source {
            Some(SourceItem::App(process)) => {
                // Try to find per-source stats for this PID
                if let Some(source_stat) = self
                    .stats
                    .per_source_stats
                    .iter()
                    .find(|s| s.pid == process.pid)
                {
                    return (source_stat.left_rms_db + source_stat.right_rms_db) / 2.0;
                }
            }
            Some(SourceItem::Microphone(_)) => {
                // Microphone stats come through combined RMS, fall through to fallback
            }
            None => {}
        }
        // Fallback to combined level when per-source stats unavailable
        (self.stats.left_rms_db + self.stats.right_rms_db) / 2.0
    }

    /// Render a horizontal level meter
    pub(crate) fn render_level_meter(&self, level_db: f32) -> impl IntoElement {
        // Map -60dB to 0dB => 0.0 to 1.0
        let normalized = ((level_db + 60.0) / 60.0).clamp(0.0, 1.0);

        // Meter bar container
        gpui::div()
            .w_full()
            .h(px(4.0))
            .rounded(px(2.0))
            .bg(colors::bg_secondary())
            .overflow_hidden()
            .child(
                // Filled portion - always green
                gpui::div()
                    .h_full()
                    .w(relative(normalized))
                    .rounded(px(2.0))
                    .bg(colors::success()),
            )
    }
}
