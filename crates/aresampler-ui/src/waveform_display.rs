use crate::app::colors;
use crate::playback::{load_samples_for_region, AudioPlayer};
use crate::waveform::{DragHandle, TrimSelection, WaveformData, WaveformView};
use gpui::{
    div, point, prelude::FluentBuilder, px, Bounds, Context, CursorStyle, InteractiveElement,
    IntoElement, MouseDownEvent, MouseMoveEvent, ParentElement, Pixels, Point, Size, Styled,
    Window,
};
use gpui_component::{h_flex, v_flex};
use std::ops::DerefMut;
use std::path::PathBuf;
use std::sync::Arc;

/// Trait that provides access to waveform-related state.
/// Implement this trait for your app state to use the waveform display component.
pub trait WaveformDisplayState {
    fn waveform_data(&self) -> Option<&Arc<WaveformData>>;
    fn trim_selection(&self) -> &TrimSelection;
    fn trim_selection_mut(&mut self) -> &mut TrimSelection;
    fn dragging_handle(&self) -> Option<DragHandle>;
    fn set_dragging_handle(&mut self, handle: Option<DragHandle>);
    fn waveform_bounds(&self) -> Option<Bounds<Pixels>>;
    fn set_waveform_bounds(&mut self, bounds: Option<Bounds<Pixels>>);
    fn is_playing(&self) -> bool;
    fn set_is_playing(&mut self, playing: bool);
    fn output_path(&self) -> Option<&PathBuf>;
    fn audio_player(&self) -> Option<&AudioPlayer>;
    fn set_error_message(&mut self, msg: Option<String>);
}

/// Handle mouse down on the waveform for trim handle interaction
pub fn handle_waveform_mouse_down<T: WaveformDisplayState>(
    state: &mut T,
    position: Point<Pixels>,
    cx: &mut Context<T>,
) {
    let Some(bounds) = state.waveform_bounds() else {
        return;
    };

    let x: f32 = position.x.into();
    let origin_x: f32 = bounds.origin.x.into();
    let width: f32 = bounds.size.width.into();

    if width <= 0.0 {
        return;
    }

    let normalized = ((x - origin_x) / width).clamp(0.0, 1.0);

    // Determine which handle to drag based on proximity
    let start_x = state.trim_selection().start;
    let end_x = state.trim_selection().end;

    let dist_to_start = (normalized - start_x).abs();
    let dist_to_end = (normalized - end_x).abs();

    // Use a hit threshold of 3% of the width
    let threshold = 0.03;

    if dist_to_start < threshold && dist_to_start <= dist_to_end {
        state.set_dragging_handle(Some(DragHandle::Start));
    } else if dist_to_end < threshold {
        state.set_dragging_handle(Some(DragHandle::End));
    } else if normalized < start_x {
        // Clicked before start handle - move start handle
        state.set_dragging_handle(Some(DragHandle::Start));
        state.trim_selection_mut().start = normalized;
    } else if normalized > end_x {
        // Clicked after end handle - move end handle
        state.set_dragging_handle(Some(DragHandle::End));
        state.trim_selection_mut().end = normalized;
    } else {
        // Clicked between handles - determine closest
        if dist_to_start < dist_to_end {
            state.set_dragging_handle(Some(DragHandle::Start));
        } else {
            state.set_dragging_handle(Some(DragHandle::End));
        }
    }

    // Stop playback when handles are dragged
    if state.is_playing() {
        stop_playback(state, cx);
    }

    cx.notify();
}

/// Handle mouse move on the waveform for trim handle dragging
pub fn handle_waveform_mouse_move<T: WaveformDisplayState>(
    state: &mut T,
    position: Point<Pixels>,
    cx: &mut Context<T>,
) {
    let Some(handle) = state.dragging_handle() else {
        return;
    };

    let Some(bounds) = state.waveform_bounds() else {
        return;
    };

    let x: f32 = position.x.into();
    let origin_x: f32 = bounds.origin.x.into();
    let width: f32 = bounds.size.width.into();

    if width <= 0.0 {
        return;
    }

    let normalized = ((x - origin_x) / width).clamp(0.0, 1.0);

    // Minimum gap between handles (2%)
    let min_gap = 0.02;

    match handle {
        DragHandle::Start => {
            let end = state.trim_selection().end;
            state.trim_selection_mut().start = normalized.min(end - min_gap);
        }
        DragHandle::End => {
            let start = state.trim_selection().start;
            state.trim_selection_mut().end = normalized.max(start + min_gap);
        }
    }

    cx.notify();
}

/// Handle mouse up on the waveform to stop dragging
pub fn handle_waveform_mouse_up<T: WaveformDisplayState>(state: &mut T, cx: &mut Context<T>) {
    state.set_dragging_handle(None);
    cx.notify();
}

/// Toggle playback state
pub fn toggle_playback<T: WaveformDisplayState>(state: &mut T, cx: &mut Context<T>) {
    if state.is_playing() {
        stop_playback(state, cx);
    } else {
        start_playback(state, cx);
    }
}

/// Start audio playback for the selected region
pub fn start_playback<T: WaveformDisplayState>(state: &mut T, cx: &mut Context<T>) {
    let Some(path) = state.output_path().cloned() else {
        state.set_error_message(Some("No output file to play".into()));
        cx.notify();
        return;
    };

    let Some(player) = state.audio_player() else {
        state.set_error_message(Some("Audio playback not available".into()));
        cx.notify();
        return;
    };

    let start = state.trim_selection().start;
    let end = state.trim_selection().end;

    // Load samples for the selected region
    match load_samples_for_region(&path, start, end) {
        Ok((samples, sample_rate, channels)) => {
            player.play_samples(samples, sample_rate, channels);
            state.set_is_playing(true);
            state.set_error_message(None);

            // Schedule a check to detect when playback finishes
            cx.spawn(async move |this, mut cx| loop {
                cx.background_executor()
                    .timer(std::time::Duration::from_millis(100))
                    .await;
                let should_stop = this
                    .update(cx.deref_mut(), |state, _cx| {
                        if let Some(player) = state.audio_player() {
                            player.is_empty()
                        } else {
                            true
                        }
                    })
                    .unwrap_or(true);

                if should_stop {
                    let _ = this.update(cx.deref_mut(), |state, cx| {
                        state.set_is_playing(false);
                        cx.notify();
                    });
                    break;
                }
            })
            .detach();
        }
        Err(e) => {
            state.set_error_message(Some(format!("Failed to load audio: {}", e)));
        }
    }
    cx.notify();
}

/// Stop audio playback
pub fn stop_playback<T: WaveformDisplayState>(state: &mut T, cx: &mut Context<T>) {
    if let Some(player) = state.audio_player() {
        player.stop();
    }
    state.set_is_playing(false);
    cx.notify();
}

/// Render the waveform section with time display and controls.
/// Returns an element that can be used in a GPUI view.
///
/// # Arguments
/// * `state` - The state implementing WaveformDisplayState
/// * `data` - The waveform data to display
/// * `on_cut` - Callback when the Cut Selection button is clicked
/// * `on_reset` - Callback when the Reset button is clicked
/// * `cx` - The GPUI context
pub fn render_waveform_section<T, OnCut, OnReset>(
    state: &T,
    data: Arc<WaveformData>,
    on_cut: OnCut,
    on_reset: OnReset,
    cx: &mut Context<T>,
) -> impl IntoElement
where
    T: WaveformDisplayState + 'static,
    OnCut: Fn(&mut T, &mut Window, &mut Context<T>) + 'static,
    OnReset: Fn(&mut T, &mut Window, &mut Context<T>) + 'static,
{
    let trim_selection = state.trim_selection().clone();
    let is_modified = state.trim_selection().is_modified();
    let is_playing = state.is_playing();
    let duration = data.duration_secs;

    // Calculate times based on trim selection
    let start_time = format!(
        "{}:{:02}",
        (duration * state.trim_selection().start as f64) as u32 / 60,
        (duration * state.trim_selection().start as f64) as u32 % 60
    );
    let current_time = format!(
        "{}:{:02}",
        (duration * 0.75) as u32 / 60,
        (duration * 0.75) as u32 % 60
    );
    let end_time = format!(
        "{}:{:02}",
        (duration * state.trim_selection().end as f64) as u32 / 60,
        (duration * state.trim_selection().end as f64) as u32 % 60
    );

    v_flex()
        .px_4()
        .py_3()
        .gap_3()
        // Waveform container
        .child(
            div()
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
                        this.set_waveform_bounds(Some(bounds));
                        handle_waveform_mouse_down(this, event.position, cx);
                    }),
                )
                .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _window, cx| {
                    handle_waveform_mouse_move(this, event.position, cx);
                }))
                .on_mouse_up(
                    gpui::MouseButton::Left,
                    cx.listener(|this, _, _window, cx| {
                        handle_waveform_mouse_up(this, cx);
                    }),
                )
                .child(
                    WaveformView::new(data)
                        .with_trim_selection(trim_selection)
                        .render(),
                ),
        )
        // Time display row
        .child(
            h_flex()
                .justify_between()
                .text_xs()
                .text_color(colors::text_muted())
                .child(start_time)
                .child(div().text_color(colors::accent()).child(current_time))
                .child(end_time),
        )
        // Control buttons
        .child(
            h_flex()
                .gap_2()
                // Play button (icon only)
                .child(
                    div()
                        .id("play-button")
                        .size(px(44.0))
                        .rounded(px(10.0))
                        .bg(colors::bg_tertiary())
                        .border_1()
                        .border_color(colors::border())
                        .flex()
                        .items_center()
                        .justify_center()
                        .cursor_pointer()
                        .on_mouse_down(
                            gpui::MouseButton::Left,
                            cx.listener(|this, _, _window, cx| {
                                toggle_playback(this, cx);
                            }),
                        )
                        .child(
                            div()
                                .text_color(colors::text_secondary())
                                .child(if is_playing { "⏹" } else { "▶" }),
                        ),
                )
                // Cut Selection button
                .child(
                    div()
                        .id("cut-button")
                        .flex_1()
                        .py_3()
                        .rounded(px(10.0))
                        .bg(colors::bg_tertiary())
                        .border_1()
                        .border_color(colors::border())
                        .flex()
                        .items_center()
                        .justify_center()
                        .cursor_pointer()
                        .text_sm()
                        .text_color(colors::text_secondary())
                        .when(!is_modified, |this| this.opacity(0.5).cursor_not_allowed())
                        .when(is_modified, |this| {
                            this.on_mouse_down(
                                gpui::MouseButton::Left,
                                cx.listener(move |this, _, window, cx| {
                                    on_cut(this, window, cx);
                                }),
                            )
                        })
                        .child("Cut Selection"),
                )
                // Reset button (icon only, same size as play button)
                .child(
                    div()
                        .id("reset-button")
                        .size(px(44.0))
                        .rounded(px(10.0))
                        .bg(colors::bg_tertiary())
                        .border_1()
                        .border_color(colors::border())
                        .flex()
                        .items_center()
                        .justify_center()
                        .cursor_pointer()
                        .on_mouse_down(
                            gpui::MouseButton::Left,
                            cx.listener(move |this, _, window, cx| {
                                on_reset(this, window, cx);
                            }),
                        )
                        .child(div().text_color(colors::text_secondary()).child("↺")),
                ),
        )
}
