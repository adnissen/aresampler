use crate::playback::{load_samples_for_region, AudioPlayer};
use crate::waveform::{trim_wav_file, DragHandle, TrimSelection, WaveformData, WaveformView};
use aresampler_core::{
    enumerate_audio_sessions, is_capture_available, request_capture_permission, AudioSessionInfo,
    CaptureConfig, CaptureEvent, CaptureSession, CaptureStats, MonitorConfig, PermissionStatus,
};
use gpui::prelude::FluentBuilder;
use gpui::*;
use gpui_component::{
    button::{Button, ButtonVariants},
    h_flex,
    select::{SearchableVec, Select, SelectEvent, SelectItem, SelectState},
    v_flex,
};
use std::collections::HashMap;
use std::ops::DerefMut;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::Instant;

// Color scheme
mod colors {
    use gpui::{rgb, Rgba};

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

// Newtype wrapper to implement SelectItem (orphan rule workaround)
#[derive(Clone)]
pub struct ProcessItem {
    pub pid: u32,
    pub name: String,
    /// Pre-rendered icon for GPUI display
    pub icon: Option<Arc<RenderImage>>,
}

impl ProcessItem {
    /// Create a ProcessItem from AudioSessionInfo, using the icon cache for efficiency.
    /// If the icon is not in the cache, it will be decoded and added.
    fn from_audio_session(
        info: AudioSessionInfo,
        icon_cache: &mut HashMap<String, Arc<RenderImage>>,
    ) -> Self {
        // Use bundle_id as cache key, fall back to name if no bundle_id
        let cache_key = info.bundle_id.clone().unwrap_or_else(|| info.name.clone());

        // Check cache first
        let icon = if let Some(cached_icon) = icon_cache.get(&cache_key) {
            Some(cached_icon.clone())
        } else {
            // Not in cache - decode and cache it
            let new_icon = info.icon_png.and_then(|png_bytes| {
                let img = image::load_from_memory(&png_bytes).ok()?;
                let mut rgba = img.to_rgba8();

                // Convert RGBA to BGRA (GPUI expects BGRA format)
                for pixel in rgba.chunks_exact_mut(4) {
                    pixel.swap(0, 2); // Swap R and B channels
                }

                let frame = image::Frame::new(rgba);
                Some(Arc::new(RenderImage::new(vec![frame])))
            });

            // Cache the icon if we got one
            if let Some(ref icon) = new_icon {
                icon_cache.insert(cache_key, icon.clone());
            }

            new_icon
        };

        Self {
            pid: info.pid,
            name: info.name,
            icon,
        }
    }
}

impl SelectItem for ProcessItem {
    type Value = u32;

    fn value(&self) -> &Self::Value {
        &self.pid
    }

    fn title(&self) -> SharedString {
        format!("{}", self.name).into()
    }

    fn display_title(&self) -> Option<AnyElement> {
        Some(
            h_flex()
                .gap_2()
                .items_center()
                .when_some(self.icon.clone(), |this, icon| {
                    this.child(
                        img(ImageSource::Render(icon))
                            .size(px(16.0))
                            .flex_shrink_0(),
                    )
                })
                .child(div().overflow_hidden().text_ellipsis().child(self.title()))
                .into_any_element(),
        )
    }

    fn render(&self, _window: &mut Window, _cx: &mut App) -> impl IntoElement {
        h_flex()
            .gap_2()
            .items_center()
            .when_some(self.icon.clone(), |this, icon| {
                this.child(
                    img(ImageSource::Render(icon))
                        .size(px(16.0))
                        .flex_shrink_0(),
                )
            })
            .when(self.icon.is_none(), |this| {
                // Empty placeholder to maintain alignment when no icon
                this.child(div().size(px(16.0)).flex_shrink_0())
            })
            .child(div().overflow_hidden().text_ellipsis().child(self.title()))
    }
}

pub struct AppState {
    // Permission state
    has_permission: bool,
    permission_error: Option<String>,

    // Process selection
    processes: Vec<ProcessItem>,
    select_state: Entity<SelectState<SearchableVec<ProcessItem>>>,
    selected_process: Option<ProcessItem>,

    // Icon cache: maps bundle_id (or name) to rendered icon
    icon_cache: HashMap<String, Arc<RenderImage>>,

    // Output file
    output_path: Option<PathBuf>,

    // Pre-roll / Monitoring state
    pre_roll_seconds: f32,
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

        // Initialize icon cache and enumerate processes
        let mut icon_cache = HashMap::new();
        let processes: Vec<ProcessItem> = if has_permission {
            enumerate_audio_sessions()
                .unwrap_or_default()
                .into_iter()
                .map(|info| ProcessItem::from_audio_session(info, &mut icon_cache))
                .collect()
        } else {
            Vec::new()
        };
        let searchable = SearchableVec::new(processes.clone());

        // Create select state
        let select_state = cx.new(|cx| SelectState::new(searchable, None, window, cx));

        // Subscribe to select events
        cx.subscribe(
            &select_state,
            |this, _, event: &SelectEvent<SearchableVec<ProcessItem>>, cx| {
                if let SelectEvent::Confirm(Some(pid)) = event {
                    // Find the selected process by PID
                    if let Some(process) = this.processes.iter().find(|p| p.pid == *pid) {
                        let process_clone = process.clone();
                        this.selected_process = Some(process_clone.clone());
                        this.error_message = None;

                        // Start monitoring for this process if pre-roll is enabled
                        if this.pre_roll_seconds > 0.0 {
                            this.start_monitoring(&process_clone, cx);
                        }

                        cx.notify();
                    }
                }
            },
        )
        .detach();

        Self {
            has_permission,
            permission_error,
            processes,
            select_state,
            selected_process: None,
            icon_cache,
            output_path: None,
            pre_roll_seconds: 10.0,
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

        // Use the icon cache to avoid re-decoding icons for known processes
        self.processes = enumerate_audio_sessions()
            .unwrap_or_default()
            .into_iter()
            .map(|info| ProcessItem::from_audio_session(info, &mut self.icon_cache))
            .collect();
        let searchable = SearchableVec::new(self.processes.clone());

        // Update select state with new items
        self.select_state.update(cx, |state, _cx| {
            state.set_items(searchable, window, _cx);
        });

        self.selected_process = None;
        cx.notify();
    }

    fn start_monitoring(&mut self, process: &ProcessItem, cx: &mut Context<Self>) {
        // Stop any existing monitoring/recording session
        self.stop_monitoring(cx);

        let config = MonitorConfig {
            pid: process.pid,
            pre_roll_duration_secs: self.pre_roll_seconds,
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

        let Some(process) = &self.selected_process else {
            self.error_message = Some("Please select a process".into());
            cx.notify();
            return;
        };

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
        let config = CaptureConfig {
            pid: process.pid,
            output_path: path.clone(),
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
                cx.spawn(async move |this, mut cx| loop {
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

    fn reset_session(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // Stop any active playback
        if self.is_playing {
            self.stop_playback(cx);
        }

        // Clear source selection
        self.selected_process = None;
        self.select_state.update(cx, |state, cx| {
            state.set_selected_index(None, window, cx);
        });

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
                .child(self.render_header())
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

        let can_record =
            self.selected_process.is_some() && self.output_path.is_some() && !self.is_recording;

        let record_button_label = if self.is_recording {
            "Stop Recording"
        } else {
            "Start Recording"
        };

        // Pre-roll values for the segmented toggle
        let pre_roll_options: [(f32, &str); 4] =
            [(0.0, "Off"), (5.0, "5s"), (10.0, "10s"), (30.0, "30s")];

        v_flex()
            .size_full()
            .bg(colors::bg_secondary())
            .text_color(colors::text_primary())
            // Header with logo
            .child(self.render_header())
            // Main content
            .child(
                v_flex()
                    .flex_1()
                    .gap_0()
                    // Source Selection Card
                    .child(self.render_source_card(cx))
                    // Arrow divider between source and output
                    .child(self.render_arrow_divider())
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
                                .border_color(colors::border())
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(colors::text_secondary())
                                        .child("Pre-roll buffer"),
                                )
                                .child(
                                    h_flex()
                                        .gap_1()
                                        .p_1()
                                        .bg(colors::bg_tertiary())
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
                                                        this.bg(colors::bg_secondary())
                                                            .text_color(colors::accent())
                                                    })
                                                    .when(!is_active, |this| {
                                                        this.text_color(colors::text_muted())
                                                    })
                                                    .when(is_disabled, |this| {
                                                        this.opacity(0.5).cursor_not_allowed()
                                                    })
                                                    .when(!is_disabled, |this| {
                                                        this.on_mouse_down(
                                                            gpui::MouseButton::Left,
                                                            cx.listener(
                                                                move |this, _, _window, cx| {
                                                                    if value == 0.0
                                                                        && this.is_monitoring
                                                                    {
                                                                        // Turning off pre-roll - stop monitoring
                                                                        this.stop_monitoring(cx);
                                                                    } else if this.is_monitoring {
                                                                        // Already monitoring - resize the buffer
                                                                        if let Some(session) =
                                                                            &mut this.capture_session
                                                                        {
                                                                            let _ = session
                                                                                .resize_pre_roll(
                                                                                    value,
                                                                                );
                                                                        }
                                                                    } else if value > 0.0 {
                                                                        // Not monitoring - start monitoring
                                                                        if let Some(process) = this
                                                                            .selected_process
                                                                            .clone()
                                                                        {
                                                                            this.start_monitoring(
                                                                                &process, cx,
                                                                            );
                                                                        }
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
                        this.child(self.render_stats_row())
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
                    }),
            )
            .into_any_element()
    }
}

impl AppState {
    /// Render the header with app name (matches GPUI TitleBar: 34px height, 80px left padding)
    fn render_header(&self) -> impl IntoElement {
        h_flex()
            .w_full()
            .h(px(34.0)) // GPUI TITLE_BAR_HEIGHT
            .pl(px(80.0)) // GPUI TITLE_BAR_LEFT_PADDING for macOS traffic lights
            .pr_4()
            .items_center()
            .border_b_1()
            .border_color(colors::border())
            .child(
                div()
                    .text_sm()
                    .font_weight(FontWeight::SEMIBOLD)
                    .child("Aresampler"),
            )
    }

    /// Render the arrow divider between source and output cards
    fn render_arrow_divider(&self) -> impl IntoElement {
        h_flex()
            .w_full()
            .px_4()
            .items_center()
            .gap_3()
            // Left line segment
            .child(
                div()
                    .flex_1()
                    .h(px(1.0))
                    .bg(colors::border()),
            )
            // Arrow icon in the middle
            .child(
                div()
                    .text_sm()
                    .text_color(colors::text_muted())
                    .child("↓"),
            )
            // Right line segment
            .child(
                div()
                    .flex_1()
                    .h(px(1.0))
                    .bg(colors::border()),
            )
    }

    /// Render the source (application) selection card
    fn render_source_card(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let has_selection = self.selected_process.is_some();

        div()
            .px_4()
            .py_3()
            .child(
                // Container with relative positioning for the overlay
                div()
                    .relative()
                    .w_full()
                    .child(
                        // Custom styled card (visual only)
                        div()
                            .w_full()
                            .px_3()
                            .py_2()
                            .rounded(px(10.0))
                            .when(has_selection, |this| this.bg(colors::bg_tertiary()))
                            .when(!has_selection, |this| {
                                this.border_1().border_color(colors::border())
                            })
                            .child(
                                h_flex()
                                    .gap_3()
                                    .items_center()
                                    // Icon
                                    .child(if let Some(process) = &self.selected_process {
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
                                            self.render_placeholder_icon().into_any_element()
                                        }
                                    } else {
                                        self.render_placeholder_icon().into_any_element()
                                    })
                                    // Text content
                                    .child(
                                        v_flex()
                                            .flex_1()
                                            .gap(px(2.0))
                                            .child(
                                                div()
                                                    .text_xs()
                                                    .text_color(colors::text_muted())
                                                    .child("RECORDING FROM"),
                                            )
                                            .child(
                                                div()
                                                    .text_sm()
                                                    .font_weight(FontWeight::MEDIUM)
                                                    .when(!has_selection, |this| {
                                                        this.text_color(colors::text_muted())
                                                    })
                                                    .child(
                                                        self.selected_process
                                                            .as_ref()
                                                            .map(|p| p.name.clone())
                                                            .unwrap_or_else(|| {
                                                                "Select application...".to_string()
                                                            }),
                                                    ),
                                            ),
                                    )
                                    // Chevron
                                    .child(div().text_color(colors::text_muted()).child("▼")),
                            ),
                    )
                    // Invisible Select overlay (handles click and dropdown)
                    .child(
                        div()
                            .id("source-select-overlay")
                            .absolute()
                            .top_0()
                            .left_0()
                            .w_full()
                            .h_full()
                            .cursor_pointer()
                            .on_mouse_down(
                                gpui::MouseButton::Left,
                                cx.listener(|this, _, window, cx| {
                                    // Refresh processes before the dropdown opens
                                    this.refresh_processes(window, cx);
                                }),
                            )
                            .child(
                                Select::new(&self.select_state)
                                    .w_full()
                                    .h_full()
                                    .appearance(false)
                                    .opacity(0.0)
                                    .placeholder("Select application..."),
                            ),
                    ),
            )
    }

    /// Render a placeholder icon for empty selection
    fn render_placeholder_icon(&self) -> impl IntoElement {
        div()
            .size(px(28.0))
            .rounded(px(6.0))
            .bg(colors::bg_tertiary())
            .border_1()
            .border_color(colors::border())
            .flex()
            .items_center()
            .justify_center()
            .child(
                div()
                    .size(px(14.0))
                    .rounded(px(3.0))
                    .border_1()
                    .border_color(colors::text_muted()),
            )
    }

    /// Render the output file selection card
    fn render_output_card(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let has_output = self.output_path.is_some();
        let output_name = self
            .output_path
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "Choose destination...".to_string());

        div()
            .px_4()
            .py_3()
            .border_b_1()
            .border_color(colors::border())
            .child(
                div()
                    .id("output-card")
                    .w_full()
                    .px_3()
                    .py_2()
                    .rounded(px(10.0))
                    .cursor_pointer()
                    .when(has_output, |this| this.bg(colors::bg_tertiary()))
                    .when(!has_output, |this| {
                        this.border_1().border_color(colors::border())
                    })
                    .on_mouse_down(
                        gpui::MouseButton::Left,
                        cx.listener(|this, _, window, cx| {
                            this.browse_output(window, cx);
                        }),
                    )
                    .child(
                        h_flex()
                            .gap_3()
                            .items_center()
                            // File icon
                            .child(self.render_file_icon(has_output))
                            // Text content
                            .child(
                                v_flex()
                                    .flex_1()
                                    .gap(px(2.0))
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(colors::text_muted())
                                            .child("SAVE TO"),
                                    )
                                    .child(
                                        div()
                                            .text_sm()
                                            .font_weight(FontWeight::MEDIUM)
                                            .when(!has_output, |this| {
                                                this.text_color(colors::text_muted())
                                            })
                                            .child(output_name),
                                    ),
                            )
                            // Chevron
                            .child(div().text_color(colors::text_muted()).child("▼")),
                    ),
            )
    }

    /// Render the file icon (purple gradient when selected, gray when empty)
    fn render_file_icon(&self, has_output: bool) -> impl IntoElement {
        div()
            .size(px(28.0))
            .rounded(px(6.0))
            .flex()
            .items_center()
            .justify_center()
            .when(has_output, |this| this.bg(colors::file_icon()))
            .when(!has_output, |this| this.bg(colors::bg_tertiary()))
            .child(
                div()
                    .text_xs()
                    .text_color(if has_output {
                        rgb(0xffffff)
                    } else {
                        colors::text_muted()
                    })
                    .child("📄"),
            )
    }

    /// Render the horizontal stats row
    fn render_stats_row(&self) -> impl IntoElement {
        // Total duration includes any captured pre-roll
        let total_duration = self.stats.duration_secs + self.captured_pre_roll_secs as f64;
        let duration = format!("{:.1}s", total_duration);
        let size = format!(
            "{:.1} MB",
            self.stats.file_size_bytes as f64 / (1024.0 * 1024.0)
        );
        // Average the left and right RMS for a single level display
        let level = (self.stats.left_rms_db + self.stats.right_rms_db) / 2.0;
        let level_str = format!("{:.0} dB", level);

        h_flex()
            .px_4()
            .py_3()
            .gap_2()
            .border_b_1()
            .border_color(colors::border())
            // Duration stat
            .child(
                v_flex()
                    .flex_1()
                    .items_center()
                    .p_2()
                    .bg(colors::bg_tertiary())
                    .rounded_md()
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::MEDIUM)
                            .child(duration),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(colors::text_muted())
                            .child("Duration"),
                    ),
            )
            // Size stat
            .child(
                v_flex()
                    .flex_1()
                    .items_center()
                    .p_2()
                    .bg(colors::bg_tertiary())
                    .rounded_md()
                    .child(div().text_sm().font_weight(FontWeight::MEDIUM).child(size))
                    .child(
                        div()
                            .text_xs()
                            .text_color(colors::text_muted())
                            .child("Size"),
                    ),
            )
            // Level stat
            .child(
                v_flex()
                    .flex_1()
                    .items_center()
                    .p_2()
                    .bg(colors::bg_tertiary())
                    .rounded_md()
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(colors::success())
                            .child(level_str),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(colors::text_muted())
                            .child("Level"),
                    ),
            )
    }

    /// Render the waveform section with time display and controls
    fn render_waveform_section(
        &self,
        data: Arc<WaveformData>,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
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
                                cx.listener(|this, _, window, cx| {
                                    this.toggle_playback(window, cx);
                                }),
                            )
                            .child(
                                div()
                                    .text_color(colors::text_secondary())
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
                            .bg(colors::bg_tertiary())
                            .border_1()
                            .border_color(colors::border())
                            .flex()
                            .items_center()
                            .justify_center()
                            .cursor_pointer()
                            .on_mouse_down(
                                gpui::MouseButton::Left,
                                cx.listener(|this, _, window, cx| {
                                    this.reset_session(window, cx);
                                }),
                            )
                            .child(div().text_color(colors::text_secondary()).child("↺")),
                    ),
            )
    }
}
