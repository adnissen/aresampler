use aresampler_core::{
    enumerate_audio_sessions, is_capture_available, request_capture_permission, AudioSessionInfo,
    CaptureConfig, CaptureEvent, CaptureSession, CaptureStats, PermissionStatus,
};
use crate::waveform::{WaveformData, WaveformView};
use gpui::prelude::FluentBuilder;
use gpui::*;
use gpui_component::{
    button::{Button, ButtonVariants},
    h_flex,
    select::{SearchableVec, Select, SelectEvent, SelectItem, SelectState},
    v_flex, Disableable,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::Instant;

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

    // Recording state
    is_recording: bool,
    stats: CaptureStats,
    capture_session: Option<CaptureSession>,
    event_receiver: Option<Receiver<CaptureEvent>>,

    // Error message
    error_message: Option<String>,

    // Last time we triggered a render update
    last_render_time: Option<Instant>,

    // Waveform data loaded after recording completes
    waveform_data: Option<Arc<WaveformData>>,
    waveform_loading: bool,
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
                        this.selected_process = Some(process.clone());
                        this.error_message = None;
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
            is_recording: false,
            stats: CaptureStats::default(),
            capture_session: None,
            event_receiver: None,
            error_message: None,
            last_render_time: None,
            waveform_data: None,
            waveform_loading: false,
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

    fn toggle_recording(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        if self.is_recording {
            self.stop_recording(cx);
        } else {
            self.start_recording(cx);
        }
    }

    fn start_recording(&mut self, cx: &mut Context<Self>) {
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
                        // Recording started successfully
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
            if this.is_recording {
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

impl Render for AppState {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Poll for capture events during render (simple approach)
        if self.is_recording {
            self.poll_capture_events();
            // Request another frame to continue polling, but only notify every 100ms
            Self::schedule_next_frame_check(window, cx);
        }

        // If we don't have permission, show permission request UI
        if !self.has_permission {
            return v_flex()
                .size_full()
                .p_4()
                .gap_4()
                .bg(rgb(0x1e1e1e))
                .text_color(rgb(0xffffff))
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
                                .text_color(rgb(0xaaaaaa))
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
                            .bg(rgb(0x5c1a1a))
                            .rounded_md()
                            .text_sm()
                            .text_color(rgb(0xff6b6b))
                            .child(msg),
                    )
                })
                .into_any_element();
        }

        let can_record =
            self.selected_process.is_some() && self.output_path.is_some() && !self.is_recording;

        let record_button_label = if self.is_recording {
            "Stop Recording"
        } else {
            "Start Recording"
        };

        let output_display = self
            .output_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "No file selected".to_string());

        v_flex()
            .size_full()
            .p_4()
            .gap_4()
            .bg(rgb(0x1e1e1e))
            .text_color(rgb(0xffffff))
            // Process selector section
            .child(
                v_flex()
                    .gap_2()
                    .child(div().text_sm().child("Select Application:"))
                    .child(
                        div()
                            .on_mouse_down(
                                gpui::MouseButton::Left,
                                cx.listener(|this, _, window, cx| {
                                    this.refresh_processes(window, cx);
                                }),
                            )
                            .child(
                                Select::new(&self.select_state)
                                    .w_full()
                                    .placeholder("Select an application..."),
                            ),
                    ),
            )
            // Output file section
            .child(
                v_flex()
                    .gap_2()
                    .child(div().text_sm().child("Output File:"))
                    .child(
                        h_flex()
                            .gap_2()
                            .child(
                                div()
                                    .flex_1()
                                    .px_2()
                                    .py_1()
                                    .bg(rgb(0x2d2d2d))
                                    .rounded_md()
                                    .text_sm()
                                    .child(output_display),
                            )
                            .child(
                                Button::new("browse")
                                    .label("Browse...")
                                    .on_click(cx.listener(|this, _, window, cx| {
                                        this.browse_output(window, cx);
                                    })),
                            ),
                    ),
            )
            // Record button
            .child(
                Button::new("record")
                    .label(record_button_label)
                    .primary()
                    .disabled(!can_record && !self.is_recording)
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.toggle_recording(window, cx);
                    })),
            )
            // Stats display
            .child(
                v_flex()
                    .gap_1()
                    .p_3()
                    .bg(rgb(0x2d2d2d))
                    .rounded_md()
                    .child(
                        div()
                            .text_sm()
                            .text_color(rgb(0x888888))
                            .child("Recording Stats:"),
                    )
                    .child(
                        div()
                            .text_sm()
                            .child(format!("Duration: {:.1}s", self.stats.duration_secs)),
                    )
                    .child(
                        div()
                            .text_sm()
                            .child(format!("Frames: {}", self.stats.total_frames)),
                    )
                    .child(div().text_sm().child(format!(
                        "Size: {:.2} MB",
                        self.stats.file_size_bytes as f64 / (1024.0 * 1024.0)
                    )))
                    .child(
                        div()
                            .text_sm()
                            .child(format!("Buffer: {} frames", self.stats.buffer_frames)),
                    )
                    .child(
                        div()
                            .text_sm()
                            .child(format!("Left: {:.1} dB", self.stats.left_rms_db)),
                    )
                    .child(
                        div()
                            .text_sm()
                            .child(format!("Right: {:.1} dB", self.stats.right_rms_db)),
                    ),
            )
            // Waveform display (show after recording completes)
            .when_some(self.waveform_data.clone(), |this, data| {
                this.child(
                    v_flex()
                        .gap_2()
                        .child(div().text_sm().child("Waveform:"))
                        .child(
                            div()
                                .h(px(100.0))
                                .w_full()
                                .rounded_md()
                                .overflow_hidden()
                                .child(WaveformView::new(data).render()),
                        ),
                )
            })
            .when(self.waveform_loading, |this| {
                this.child(
                    div()
                        .p_2()
                        .text_sm()
                        .text_color(rgb(0x888888))
                        .child("Loading waveform..."),
                )
            })
            // Error message
            .when_some(self.error_message.clone(), |this, msg| {
                this.child(
                    div()
                        .p_2()
                        .bg(rgb(0x5c1a1a))
                        .rounded_md()
                        .text_sm()
                        .text_color(rgb(0xff6b6b))
                        .child(msg),
                )
            })
            .into_any_element()
    }
}
