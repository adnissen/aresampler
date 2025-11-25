use aresampler_core::{
    enumerate_audio_sessions, AudioSessionInfo, CaptureConfig, CaptureEvent,
    CaptureSession, CaptureStats,
};
use gpui::prelude::FluentBuilder;
use gpui::*;
use gpui_component::{
    button::{Button, ButtonVariants},
    select::{Select, SelectEvent, SelectItem, SelectState, SearchableVec},
    h_flex, v_flex, Disableable,
};
use std::path::PathBuf;
use std::sync::mpsc::Receiver;

// Newtype wrapper to implement SelectItem (orphan rule workaround)
#[derive(Clone, Debug)]
pub struct ProcessItem {
    pub process_id: u32,
    pub process_name: String,
}

impl From<AudioSessionInfo> for ProcessItem {
    fn from(info: AudioSessionInfo) -> Self {
        Self {
            process_id: info.process_id,
            process_name: info.process_name,
        }
    }
}

impl SelectItem for ProcessItem {
    type Value = u32;

    fn value(&self) -> &Self::Value {
        &self.process_id
    }

    fn title(&self) -> SharedString {
        format!("{} (PID: {})", self.process_name, self.process_id).into()
    }
}

pub struct AppState {
    // Process selection
    processes: Vec<ProcessItem>,
    select_state: Entity<SelectState<SearchableVec<ProcessItem>>>,
    selected_process: Option<ProcessItem>,

    // Output file
    output_path: Option<PathBuf>,

    // Recording state
    is_recording: bool,
    stats: CaptureStats,
    capture_session: Option<CaptureSession>,
    event_receiver: Option<Receiver<CaptureEvent>>,

    // Error message
    error_message: Option<String>,
}

impl AppState {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        // Note: We don't call initialize_audio() here because GPUI initializes COM in STA mode,
        // but WASAPI requires MTA. The capture thread handles its own COM initialization.

        // Enumerate processes with audio sessions
        let processes: Vec<ProcessItem> = enumerate_audio_sessions()
            .unwrap_or_default()
            .into_iter()
            .map(ProcessItem::from)
            .collect();
        let searchable = SearchableVec::new(processes.clone());

        // Create select state
        let select_state = cx.new(|cx| SelectState::new(searchable, None, window, cx));

        // Subscribe to select events
        cx.subscribe(&select_state, |this, _, event: &SelectEvent<SearchableVec<ProcessItem>>, cx| {
            if let SelectEvent::Confirm(Some(pid)) = event {
                // Find the selected process by PID
                if let Some(process) = this.processes.iter().find(|p| p.process_id == *pid) {
                    this.selected_process = Some(process.clone());
                    this.error_message = None;
                    cx.notify();
                }
            }
        })
        .detach();

        Self {
            processes,
            select_state,
            selected_process: None,
            output_path: None,
            is_recording: false,
            stats: CaptureStats::default(),
            capture_session: None,
            event_receiver: None,
            error_message: None,
        }
    }

    fn browse_output(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        // Use a simple text input for file path since GPUI file dialogs have API differences
        // For now, set a default path - in production you'd use platform-specific file dialog
        let default_path = std::env::current_dir()
            .unwrap_or_default()
            .join("recording.wav");
        self.output_path = Some(default_path);
        self.error_message = None;
        cx.notify();
    }

    fn refresh_processes(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.processes = enumerate_audio_sessions()
            .unwrap_or_default()
            .into_iter()
            .map(ProcessItem::from)
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
            pid: process.process_id,
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
        cx.notify();
    }
}

impl Render for AppState {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Poll for capture events during render (simple approach)
        if self.is_recording {
            self.poll_capture_events();
            // Request another frame to continue polling
            cx.notify();
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
                        h_flex()
                            .gap_2()
                            .child(
                                Select::new(&self.select_state)
                                    .w_full()
                                    .placeholder("Select an application...")
                            )
                            .child(
                                Button::new("refresh")
                                    .label("Refresh")
                                    .on_click(cx.listener(|this, _, window, cx| {
                                        this.refresh_processes(window, cx);
                                    })),
                            ),
                    ),
            )
            // Output file section
            .child(
                v_flex()
                    .gap_2()
                    .child(div().text_sm().child("Output File:"))
                    .child(
                        h_flex().gap_2().child(
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
                    ),
            )
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
    }
}
