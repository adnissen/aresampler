//! Application lifecycle: initialization, permissions, file browsing, frame scheduling

use super::playback::AudioPlayer;
use super::waveform::TrimSelection;
use super::AppState;
use crate::core::{is_capture_available, request_capture_permission, PermissionStatus};
use crate::source_selection::SourceSelectionState;
use crate::core::CaptureStats;
use gpui::{Context, Window};
use std::time::Instant;

impl AppState {
    /// Create a new AppState instance
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        // Note: We don't call initialize_audio() here because GPUI initializes COM in STA mode,
        // but WASAPI requires MTA. The capture thread handles its own COM initialization.

        // Check permission status
        let has_permission = is_capture_available().unwrap_or(false);
        let permission_error = if !has_permission {
            Some("Screen Recording permission required".to_string())
        } else {
            None
        };

        // Initialize source selection state
        let source_selection = SourceSelectionState::new(has_permission, window, cx);

        // Subscribe to the first source's select events
        Self::subscribe_to_source(&source_selection.sources[0], 0, cx);

        Self {
            has_permission,
            permission_error,
            source_selection,
            output_path: None,
            pre_roll_seconds: 0.0,
            sample_rate: 48000, // Apple recommends 48kHz for ScreenCaptureKit
            is_monitoring: false,
            is_recording: false,
            stats: CaptureStats::default(),
            captured_pre_roll_secs: 0.0,
            capture_session: None,
            event_receiver: None,
            error_message: None,
            last_render_time: None,
            waveform_data: None,
            waveform_loading: false,
            trim_selection: TrimSelection::new(),
            dragging_handle: None,
            waveform_bounds: None,
            audio_player: AudioPlayer::new().ok(),
            is_playing: false,
        }
    }

    /// Request screen recording permission
    pub(crate) fn request_permission(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        match request_capture_permission() {
            Ok(PermissionStatus::Granted) => {
                self.has_permission = true;
                self.permission_error = None;
                // Refresh processes now that we have permission
                self.refresh_processes(window, cx);
            }
            Ok(PermissionStatus::Denied) => {
                self.permission_error = Some(
                    "Permission denied. Please grant Screen Recording permission in System Preferences > Privacy & Security > Screen Recording".to_string()
                );
            }
            Ok(PermissionStatus::Unknown) => {
                self.permission_error = Some("Permission status unknown".to_string());
            }
            Err(e) => {
                self.permission_error = Some(format!("Error requesting permission: {}", e));
            }
        }
        cx.notify();
    }

    /// Refresh available processes list
    pub(crate) fn refresh_processes(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // Stop any active monitoring when refreshing
        self.stop_monitoring(cx);

        // Refresh via source selection state
        self.source_selection.refresh_processes(window, cx);
        cx.notify();
    }

    /// Browse for output file location
    pub(crate) fn browse_output(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        let directory = std::env::current_dir().unwrap_or_default();
        let receiver = cx.prompt_for_new_path(&directory, Some("recording.wav"));

        cx.spawn(async move |this, cx| {
            if let Ok(result) = receiver.await {
                if let Ok(Some(path)) = result {
                    this.update(cx, |state, cx| {
                        state.output_path = Some(path);
                        state.error_message = None;
                        cx.notify();
                    })
                    .ok();
                }
            }
        })
        .detach();
    }

    /// Schedule next frame check for UI updates during recording
    /// Throttles updates to every 100ms to avoid excessive rendering
    pub(crate) fn schedule_next_frame_check(window: &mut Window, cx: &mut Context<Self>) {
        cx.on_next_frame(window, |this, w, cx| {
            if this.is_recording || this.is_monitoring {
                let now = Instant::now();
                let should_notify = this
                    .last_render_time
                    .map(|last| now.duration_since(last).as_millis() >= 100)
                    .unwrap_or(true);
                if should_notify {
                    this.last_render_time = Some(now);
                    cx.notify();
                } else {
                    // Keep checking every frame until 100ms passes
                    Self::schedule_next_frame_check(w, cx);
                }
            }
        });
    }
}
