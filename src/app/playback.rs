//! Audio playback controls and waveform mouse interaction (trim handles)

use super::AppState;
use crate::playback::load_samples_for_region;
use crate::waveform::{DragHandle, TrimSelection, WaveformData, trim_wav_file};
use gpui::{Context, Pixels, Point, Window};
use std::ops::DerefMut;
use std::sync::Arc;

impl AppState {
    /// Toggle playback on/off
    pub(crate) fn toggle_playback(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        if self.is_playing {
            self.stop_playback(cx);
        } else {
            self.start_playback(cx);
        }
    }

    /// Start audio playback for the selected region
    pub(crate) fn start_playback(&mut self, cx: &mut Context<Self>) {
        let Some(path) = &self.output_path else {
            self.error_message = Some("No output file to play".into());
            cx.notify();
            return;
        };

        let Some(player) = &self.audio_player else {
            self.error_message = Some("Audio playback not available".into());
            cx.notify();
            return;
        };

        // Load samples for the selected region
        match load_samples_for_region(path, self.trim_selection.start, self.trim_selection.end) {
            Ok((samples, sample_rate, channels)) => {
                player.play_samples(samples, sample_rate, channels);
                self.is_playing = true;
                self.error_message = None;

                // Schedule a check to detect when playback finishes
                cx.spawn(async move |this, mut cx| {
                    loop {
                        cx.background_executor()
                            .timer(std::time::Duration::from_millis(100))
                            .await;
                        let should_stop = this
                            .update(cx.deref_mut(), |state, _cx| {
                                if let Some(player) = &state.audio_player {
                                    player.is_empty()
                                } else {
                                    true
                                }
                            })
                            .unwrap_or(true);

                        if should_stop {
                            let _ = this.update(cx.deref_mut(), |state, cx| {
                                state.is_playing = false;
                                cx.notify();
                            });
                            break;
                        }
                    }
                })
                .detach();
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to load audio: {}", e));
            }
        }
        cx.notify();
    }

    /// Stop audio playback
    pub(crate) fn stop_playback(&mut self, cx: &mut Context<Self>) {
        if let Some(player) = &self.audio_player {
            player.stop();
        }
        self.is_playing = false;
        cx.notify();
    }

    /// Cut audio to current trim selection and reload waveform
    pub(crate) fn cut_audio(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        let Some(path) = self.output_path.clone() else {
            self.error_message = Some("No output file selected".into());
            cx.notify();
            return;
        };

        // Stop any active playback first
        self.stop_playback(cx);

        let start = self.trim_selection.start;
        let end = self.trim_selection.end;

        // Show loading state
        self.waveform_loading = true;
        self.waveform_data = None;
        cx.notify();

        cx.spawn(async move |this, cx| {
            // Perform trim operation
            let trim_result = trim_wav_file(&path, start, end);

            // Then reload waveform
            let waveform_result = match &trim_result {
                Ok(()) => WaveformData::load(&path, 2000),
                Err(_) => Err(crate::waveform::WaveformError::Io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Trim failed",
                ))),
            };

            this.update(cx, |state, cx| {
                state.waveform_loading = false;

                match trim_result {
                    Ok(()) => {
                        // Reset trim selection since file changed
                        state.trim_selection = TrimSelection::new();

                        match waveform_result {
                            Ok(data) => {
                                state.waveform_data = Some(Arc::new(data));
                            }
                            Err(e) => {
                                state.error_message =
                                    Some(format!("Trim succeeded but failed to reload: {}", e));
                            }
                        }
                    }
                    Err(e) => {
                        state.error_message = Some(format!("Failed to trim audio: {}", e));
                    }
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// Handle mouse down on waveform (start dragging trim handle)
    pub(crate) fn handle_waveform_mouse_down(&mut self, position: Point<Pixels>, cx: &mut Context<Self>) {
        let Some(bounds) = self.waveform_bounds else {
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
        let start_x = self.trim_selection.start;
        let end_x = self.trim_selection.end;

        let dist_to_start = (normalized - start_x).abs();
        let dist_to_end = (normalized - end_x).abs();

        // Use a hit threshold of 3% of the width
        let threshold = 0.03;

        if dist_to_start < threshold && dist_to_start <= dist_to_end {
            self.dragging_handle = Some(DragHandle::Start);
        } else if dist_to_end < threshold {
            self.dragging_handle = Some(DragHandle::End);
        } else if normalized < start_x {
            // Clicked before start handle - move start handle
            self.dragging_handle = Some(DragHandle::Start);
            self.trim_selection.start = normalized;
        } else if normalized > end_x {
            // Clicked after end handle - move end handle
            self.dragging_handle = Some(DragHandle::End);
            self.trim_selection.end = normalized;
        } else {
            // Clicked between handles - determine closest
            if dist_to_start < dist_to_end {
                self.dragging_handle = Some(DragHandle::Start);
            } else {
                self.dragging_handle = Some(DragHandle::End);
            }
        }

        // Stop playback when handles are dragged
        if self.is_playing {
            self.stop_playback(cx);
        }

        cx.notify();
    }

    /// Handle mouse move on waveform (update trim handle position)
    pub(crate) fn handle_waveform_mouse_move(&mut self, position: Point<Pixels>, cx: &mut Context<Self>) {
        let Some(handle) = self.dragging_handle else {
            return;
        };

        let Some(bounds) = self.waveform_bounds else {
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
                self.trim_selection.start = normalized.min(self.trim_selection.end - min_gap);
            }
            DragHandle::End => {
                self.trim_selection.end = normalized.max(self.trim_selection.start + min_gap);
            }
        }

        cx.notify();
    }

    /// Handle mouse up on waveform (stop dragging)
    pub(crate) fn handle_waveform_mouse_up(&mut self, cx: &mut Context<Self>) {
        self.dragging_handle = None;
        cx.notify();
    }
}
