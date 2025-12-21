use crate::playback::{AudioPlayer, load_samples_for_region};
use crate::source_selection::{
    SourceEntry, SourceItem, SourceSelectionState, render_placeholder_icon,
};
use crate::theme::ThemeRegistry;
use crate::waveform::{DragHandle, TrimSelection, WaveformData, WaveformView, trim_wav_file};
use crate::SwitchTheme;
use crate::core::{
    CaptureConfig, CaptureEvent, CaptureSession, CaptureStats, MonitorConfig, PermissionStatus,
    is_capture_available, request_capture_permission,
};
use gpui::{
    Bounds, Context, Corner, CursorStyle, ElementId, FontWeight, ImageSource, InteractiveElement,
    IntoElement, MouseDownEvent, MouseMoveEvent, ParentElement, Pixels, Point, Render,
    SharedString, Size, StatefulInteractiveElement, Styled, Window, WindowControlArea, div, img,
    point, prelude::FluentBuilder, px, relative, rgb, svg,
};
use gpui_component::{
    Theme,
    button::{Button, ButtonVariants},
    h_flex,
    menu::DropdownMenu,
    scroll::ScrollableElement,
    select::{SearchableVec, Select, SelectEvent},
    v_flex,
};
use std::ops::DerefMut;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::Instant;

// Color scheme
pub mod colors {
    use gpui::{Rgba, rgb};

    // Backgrounds
    pub fn bg_primary() -> Rgba {
        rgb(0x0a0a0b)
    }
    pub fn bg_secondary() -> Rgba {
        rgb(0x111113)
    }
    pub fn bg_tertiary() -> Rgba {
        rgb(0x18181b)
    }

    // Borders
    pub fn border() -> Rgba {
        rgb(0x27272a)
    }

    // Text
    pub fn text_primary() -> Rgba {
        rgb(0xfafafa)
    }
    pub fn text_secondary() -> Rgba {
        rgb(0xa1a1aa)
    }
    pub fn text_muted() -> Rgba {
        rgb(0x52525b)
    }

    // Accent
    pub fn accent() -> Rgba {
        rgb(0x22d3ee)
    }

    // Recording
    pub fn recording() -> Rgba {
        rgb(0xef4444)
    }

    // Success (for level display)
    pub fn success() -> Rgba {
        rgb(0x22c55e)
    }

    // Error
    pub fn error_bg() -> Rgba {
        rgb(0x5c1a1a)
    }
    pub fn error_text() -> Rgba {
        rgb(0xff6b6b)
    }

    // File icon purple
    pub fn file_icon() -> Rgba {
        rgb(0x8b5cf6)
    }
}

pub struct AppState {
    // Permission state
    has_permission: bool,
    permission_error: Option<String>,

    // Source selection (process/application picker)
    source_selection: SourceSelectionState,

    // Output file
    output_path: Option<PathBuf>,

    // Pre-roll / Monitoring state
    pre_roll_seconds: f32,
    sample_rate: u32,
    is_monitoring: bool,

    // Recording state
    is_recording: bool,
    stats: CaptureStats,
    captured_pre_roll_secs: f32, // Pre-roll duration captured when recording started
    capture_session: Option<CaptureSession>,
    event_receiver: Option<Receiver<CaptureEvent>>,

    // Error message
    error_message: Option<String>,

    // Last time we triggered a render update
    last_render_time: Option<Instant>,

    // Waveform data loaded after recording completes
    waveform_data: Option<Arc<WaveformData>>,
    waveform_loading: bool,

    // Trim selection state
    trim_selection: TrimSelection,
    dragging_handle: Option<DragHandle>,
    waveform_bounds: Option<Bounds<Pixels>>,

    // Audio playback state
    audio_player: Option<AudioPlayer>,
    is_playing: bool,
}

impl AppState {
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

    /// Subscribe to select events for a source entry.
    fn subscribe_to_source(source: &SourceEntry, index: usize, cx: &mut Context<Self>) {
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
    fn add_source(&mut self, window: &mut Window, cx: &mut Context<Self>) {
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
    fn remove_source(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
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

    fn request_permission(&mut self, window: &mut Window, cx: &mut Context<Self>) {
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

    fn browse_output(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
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

    fn refresh_processes(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // Stop any active monitoring when refreshing
        self.stop_monitoring(cx);

        // Refresh via source selection state
        self.source_selection.refresh_processes(window, cx);
        cx.notify();
    }

    fn start_monitoring(&mut self, cx: &mut Context<Self>) {
        // Stop any existing monitoring/recording session
        self.stop_monitoring(cx);

        // Get all selected PIDs (from app sources) and microphone ID
        let pids = self.source_selection.selected_pids();
        let microphone_id = self.source_selection.selected_microphone_id();

        // Must have at least one source (app or microphone)
        if pids.is_empty() && microphone_id.is_none() {
            return;
        }

        let config = MonitorConfig {
            pids,
            sample_rate: self.sample_rate,
            pre_roll_duration_secs: self.pre_roll_seconds,
            microphone_id,
            ..Default::default()
        };

        let mut session = CaptureSession::new_empty();
        match session.start_monitoring(config) {
            Ok(rx) => {
                self.event_receiver = Some(rx);
                self.capture_session = Some(session);
                self.is_monitoring = true;
                self.is_recording = false;
                self.error_message = None;
                self.stats = CaptureStats::default();
                self.captured_pre_roll_secs = 0.0;
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to start monitoring: {}", e));
            }
        }
        cx.notify();
    }

    fn stop_monitoring(&mut self, cx: &mut Context<Self>) {
        if let Some(mut session) = self.capture_session.take() {
            let _ = session.stop();
        }
        self.is_monitoring = false;
        self.event_receiver = None;
        cx.notify();
    }

    fn toggle_recording(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        if self.is_recording {
            self.stop_recording(cx);
        } else {
            self.start_recording(cx);
        }
    }

    fn start_recording(&mut self, cx: &mut Context<Self>) {
        // Stop any active playback first
        if self.is_playing {
            self.stop_playback(cx);
        }

        // Check if any source is selected (app or microphone)
        if !self.source_selection.has_any_selection() {
            self.error_message = Some("Please select a source".into());
            cx.notify();
            return;
        }

        let Some(path) = &self.output_path else {
            self.error_message = Some("Please select an output file".into());
            cx.notify();
            return;
        };

        // Reset trim selection and waveform for new recording
        self.trim_selection = TrimSelection::new();
        self.waveform_data = None;

        // If we're monitoring, transition to recording
        if self.is_monitoring {
            if let Some(ref mut session) = self.capture_session {
                match session.start_recording(path.clone()) {
                    Ok(()) => {
                        self.is_monitoring = false;
                        self.is_recording = true;
                        self.error_message = None;
                        cx.notify();
                    }
                    Err(e) => {
                        self.error_message = Some(format!("Failed to start recording: {}", e));
                        cx.notify();
                    }
                }
            }
            return;
        }

        // Direct recording mode (no monitoring)
        let pids = self.source_selection.selected_pids();
        let microphone_id = self.source_selection.selected_microphone_id();
        let config = CaptureConfig {
            pids,
            output_path: path.clone(),
            sample_rate: self.sample_rate,
            microphone_id,
            ..Default::default()
        };

        let mut session = CaptureSession::new(config);
        match session.start() {
            Ok(rx) => {
                self.event_receiver = Some(rx);
                self.capture_session = Some(session);
                self.is_recording = true;
                self.error_message = None;
                self.stats = CaptureStats::default();
                self.captured_pre_roll_secs = 0.0; // No pre-roll in direct recording mode
                cx.notify();
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to start recording: {}", e));
                cx.notify();
            }
        }
    }

    fn poll_capture_events(&mut self) {
        let mut stopped = false;
        let mut error_msg = None;

        if let Some(rx) = &self.event_receiver {
            while let Ok(event) = rx.try_recv() {
                match event {
                    CaptureEvent::Started { .. } => {
                        // Recording started successfully (direct mode)
                    }
                    CaptureEvent::MonitoringStarted => {
                        // Monitoring started successfully
                    }
                    CaptureEvent::RecordingStarted { pre_roll_secs } => {
                        // Transitioned from monitoring to recording
                        self.is_monitoring = false;
                        self.is_recording = true;
                        self.captured_pre_roll_secs = pre_roll_secs as f32;
                    }
                    CaptureEvent::StatsUpdate(stats) => {
                        self.stats = stats;
                    }
                    CaptureEvent::Stopped => {
                        stopped = true;
                    }
                    CaptureEvent::Error(msg) => {
                        error_msg = Some(msg);
                        stopped = true;
                    }
                }
            }
        }

        if stopped {
            self.is_recording = false;
            self.is_monitoring = false;
            self.capture_session = None;
            self.event_receiver = None;
            if let Some(msg) = error_msg {
                self.error_message = Some(msg);
            }
        }
    }

    fn stop_recording(&mut self, cx: &mut Context<Self>) {
        if let Some(mut session) = self.capture_session.take() {
            let _ = session.stop();
        }
        self.is_recording = false;
        self.event_receiver = None;

        // Load waveform data after recording stops
        if let Some(path) = self.output_path.clone() {
            self.waveform_loading = true;
            cx.spawn(async move |this, cx| {
                // Load waveform in background (target ~2000 buckets)
                let result = WaveformData::load(&path, 2000);

                this.update(cx, |state, cx| {
                    state.waveform_loading = false;
                    match result {
                        Ok(data) => {
                            state.waveform_data = Some(Arc::new(data));
                        }
                        Err(e) => {
                            state.error_message = Some(format!("Failed to load waveform: {}", e));
                        }
                    }
                    cx.notify();
                })
                .ok();
            })
            .detach();
        }

        cx.notify();
    }

    fn schedule_next_frame_check(window: &mut Window, cx: &mut Context<Self>) {
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

    fn toggle_playback(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        if self.is_playing {
            self.stop_playback(cx);
        } else {
            self.start_playback(cx);
        }
    }

    fn start_playback(&mut self, cx: &mut Context<Self>) {
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

    fn stop_playback(&mut self, cx: &mut Context<Self>) {
        if let Some(player) = &self.audio_player {
            player.stop();
        }
        self.is_playing = false;
        cx.notify();
    }

    fn cut_audio(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
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

    fn reset_session(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        // Stop any active playback
        if self.is_playing {
            self.stop_playback(cx);
        }

        // Keep the current sources - don't clear selections or remove sources

        // Clear output path
        self.output_path = None;

        // Clear waveform and recording data
        self.waveform_data = None;
        self.waveform_loading = false;
        self.trim_selection = TrimSelection::new();
        self.waveform_bounds = None;

        // Reset stats
        self.stats = CaptureStats::default();
        self.captured_pre_roll_secs = 0.0;

        // Clear any error messages
        self.error_message = None;

        // Restart monitoring with the existing sources
        self.restart_monitoring(cx);

        cx.notify();
    }

    fn restart_monitoring(&mut self, cx: &mut Context<Self>) {
        // Get all selected PIDs (from app sources) and microphone ID
        let pids = self.source_selection.selected_pids();
        let microphone_id = self.source_selection.selected_microphone_id();

        // Must have at least one source (app or microphone)
        if pids.is_empty() && microphone_id.is_none() {
            return;
        }

        let config = MonitorConfig {
            pids,
            sample_rate: self.sample_rate,
            pre_roll_duration_secs: self.pre_roll_seconds,
            microphone_id,
            ..Default::default()
        };

        let mut session = CaptureSession::new_empty();
        match session.start_monitoring(config) {
            Ok(rx) => {
                self.event_receiver = Some(rx);
                self.capture_session = Some(session);
                self.is_monitoring = true;
                self.is_recording = false;
                self.error_message = None;
                self.stats = CaptureStats::default();
                self.captured_pre_roll_secs = 0.0;
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to start monitoring: {}", e));
            }
        }
        cx.notify();
    }

    fn handle_waveform_mouse_down(&mut self, position: Point<Pixels>, cx: &mut Context<Self>) {
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

    fn handle_waveform_mouse_move(&mut self, position: Point<Pixels>, cx: &mut Context<Self>) {
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

    fn handle_waveform_mouse_up(&mut self, cx: &mut Context<Self>) {
        self.dragging_handle = None;
        cx.notify();
    }
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
                                    div()
                                        .text_lg()
                                        .font_weight(FontWeight::BOLD)
                                        .child("Permission Required"),
                                )
                                .child(
                                    div()
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
                                div()
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
                                    div()
                                        .w(relative(0.5))
                                        .relative()
                                        .child(
                                            div()
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

                                                div()
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
                                    div()
                                        .w(relative(0.5))
                                        .relative()
                                        .child(
                                            div()
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

                                                div()
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
                            div().px_4().py_3().child(
                                div()
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
                                    .child(div().size(px(12.0)).rounded_full().bg(rgb(0xffffff)))
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
                            div()
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
                            div()
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
    /// Render the header with app name (34px height)
    /// On macOS: 80px left padding for traffic lights
    /// On Windows: hamburger menu + minimize/close buttons, draggable area
    fn render_header(&self, cx: &mut Context<Self>) -> impl IntoElement {
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
                        div()
                            .text_sm()
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(theme.foreground)
                            .child("aresampler"),
                    )
                    // Version number
                    .child(
                        div()
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
                            div()
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
                                    div().w(px(10.0)).h(px(1.0)).bg(colors::text_secondary()),
                                ),
                        )
                        // Close button
                        .child(
                            div()
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
                                    div()
                                        .text_xs()
                                        .text_color(colors::text_secondary())
                                        .child("✕"),
                                ),
                        ),
                )
            })
    }

    /// Render the theme palette button with dropdown theme picker (Windows only)
    fn render_hamburger_menu(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = Theme::global(cx);
        let current_theme_name = theme.theme_name().to_string();
        let icon_color = theme.foreground;

        // Build the palette button with a dropdown menu showing themes directly
        Button::new("theme-menu")
            .ghost()
            .compact()
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

    /// Render the arrow divider between source and output cards
    fn render_arrow_divider(&self, cx: &Context<Self>) -> impl IntoElement {
        let theme = Theme::global(cx);

        h_flex()
            .w_full()
            .px_4()
            .items_center()
            .gap_3()
            // Left line segment
            .child(div().flex_1().h(px(1.0)).bg(theme.border))
            // Arrow icon in the middle
            .child(
                div()
                    .text_sm()
                    .text_color(theme.muted_foreground)
                    .child("↓"),
            )
            // Right line segment
            .child(div().flex_1().h(px(1.0)).bg(theme.border))
    }

    /// Render the "+" divider between source cards
    fn render_plus_divider(&self, cx: &Context<Self>) -> impl IntoElement {
        let theme = Theme::global(cx);

        h_flex()
            .w_full()
            .px_4()
            .items_center()
            .gap_3()
            .h(px(5.0))
            // Left space segment
            .child(div().flex_1().h(px(1.0)))
            // Plus icon in the middle
            .child(
                div()
                    .text_sm()
                    .text_color(theme.muted_foreground)
                    .child("+"),
            )
            // Right space segment
            .child(div().flex_1().h(px(1.0)))
    }

    /// Render all source selection cards plus the Add Source button.
    fn render_sources_section(&self, cx: &mut Context<Self>) -> impl IntoElement {
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
                        div().px_4().pb_2().child(
                            div()
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

    /// Render a single source (application or microphone) selection card
    fn render_source_card(&self, index: usize, cx: &mut Context<Self>) -> impl IntoElement {
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

        div().px_4().py_3().child(
            // Container with relative positioning for the overlay
            div()
                .relative()
                .w_full()
                // Grey out when recording or waveform is displayed
                .when(is_locked, |this| this.opacity(0.5))
                .child(
                    // Custom styled card (visual only)
                    div()
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
                                                    div()
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
                                                div()
                                                    .size(px(28.0))
                                                    .rounded(px(6.0))
                                                    .bg(gpui::rgb(0x4a9eff))
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
                                                    div()
                                                        .text_xs()
                                                        .text_color(theme_muted_fg)
                                                        .child(label_text),
                                                )
                                                .child(
                                                    div()
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
                                                                    "Select source..."
                                                                        .to_string()
                                                                }),
                                                        ),
                                                ),
                                        )
                                        // Remove button (X) for non-first sources (hide when locked)
                                        .when(!is_first && !is_locked, |this| {
                                            this.child(
                                                div()
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
                                            this.child(div().text_color(theme_muted_fg).child("▼"))
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
                        div()
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
    fn get_source_level(&self, index: usize) -> f32 {
        let source = &self.source_selection.sources[index];
        if let Some(SourceItem::App(process)) = &source.selected_source {
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
        // Fallback to combined level (for microphones or when per-source stats unavailable)
        (self.stats.left_rms_db + self.stats.right_rms_db) / 2.0
    }

    /// Render a horizontal level meter
    fn render_level_meter(&self, level_db: f32) -> impl IntoElement {
        // Map -60dB to 0dB => 0.0 to 1.0
        let normalized = ((level_db + 60.0) / 60.0).clamp(0.0, 1.0);

        // Meter bar container
        div()
            .w_full()
            .h(px(4.0))
            .rounded(px(2.0))
            .bg(colors::bg_secondary())
            .overflow_hidden()
            .child(
                // Filled portion - always green
                div()
                    .h_full()
                    .w(relative(normalized))
                    .rounded(px(2.0))
                    .bg(colors::success()),
            )
    }

    /// Render the output file selection card
    fn render_output_card(&self, cx: &mut Context<Self>) -> impl IntoElement {
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

        div()
            .px_4()
            .py_3()
            .border_b_1()
            .border_color(theme_border)
            .child(
                div()
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
                                        div()
                                            .text_xs()
                                            .text_color(theme_muted_fg)
                                            .child(label_text),
                                    )
                                    .child(
                                        div()
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
    fn render_file_icon(&self, has_output: bool, cx: &Context<Self>) -> impl IntoElement {
        let theme = Theme::global(cx);

        div()
            .size(px(28.0))
            .rounded(px(6.0))
            .flex()
            .items_center()
            .justify_center()
            .when(has_output, |this| this.bg(theme.primary))
            .when(!has_output, |this| this.bg(theme.muted))
            .child(
                div()
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
    fn render_stats_row(&self, cx: &Context<Self>) -> impl IntoElement {
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
                        div()
                            .text_base()
                            .font_weight(FontWeight::MEDIUM)
                            .child(duration),
                    )
                    .child(
                        div()
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
                        div()
                            .text_base()
                            .font_weight(FontWeight::MEDIUM)
                            .child(size),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(theme.muted_foreground)
                            .child("Size"),
                    ),
            )
    }

    /// Render the waveform section with time display and controls
    fn render_waveform_section(
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
                    .child(div().text_color(theme.primary).child(current_time))
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
                                div()
                                    .text_color(theme.foreground)
                                    .child(if self.is_playing { "⏹" } else { "▶" }),
                            ),
                    )
                    // Cut Selection button
                    .child(
                        div()
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
                        div()
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
                            .child(div().text_color(theme.foreground).child("↺")),
                    ),
            )
    }
}
