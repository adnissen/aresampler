//! Recording and monitoring: start/stop, event polling, state transitions

use super::waveform::{TrimSelection, WaveformData};
use super::AppState;
use crate::core::{CaptureConfig, CaptureEvent, CaptureSession, CaptureStats, MonitorConfig};
use gpui::{Context, Window};
use std::sync::Arc;

impl AppState {
    /// Start audio monitoring (pre-roll buffer)
    pub(crate) fn start_monitoring(&mut self, cx: &mut Context<Self>) {
        // Stop any existing monitoring/recording session
        self.stop_monitoring(cx);

        // Get all selected PIDs (from app sources) and microphones
        let pids = self.source_selection.selected_pids();
        let microphones = self.source_selection.selected_microphones();

        // Must have at least one source (app or microphone)
        if pids.is_empty() && microphones.is_empty() {
            return;
        }

        let config = MonitorConfig {
            pids,
            sample_rate: self.sample_rate,
            pre_roll_duration_secs: self.pre_roll_seconds,
            microphones,
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

    /// Stop monitoring
    pub(crate) fn stop_monitoring(&mut self, cx: &mut Context<Self>) {
        if let Some(mut session) = self.capture_session.take() {
            let _ = session.stop();
        }
        self.is_monitoring = false;
        self.event_receiver = None;
        cx.notify();
    }

    /// Restart monitoring with current sources
    pub(crate) fn restart_monitoring(&mut self, cx: &mut Context<Self>) {
        // Get all selected PIDs (from app sources) and microphones
        let pids = self.source_selection.selected_pids();
        let microphones = self.source_selection.selected_microphones();

        // Must have at least one source (app or microphone)
        if pids.is_empty() && microphones.is_empty() {
            return;
        }

        let config = MonitorConfig {
            pids,
            sample_rate: self.sample_rate,
            pre_roll_duration_secs: self.pre_roll_seconds,
            microphones,
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

    /// Toggle recording on/off
    pub(crate) fn toggle_recording(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        if self.is_recording {
            self.stop_recording(cx);
        } else {
            self.start_recording(cx);
        }
    }

    /// Start recording
    pub(crate) fn start_recording(&mut self, cx: &mut Context<Self>) {
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
        let microphones = self.source_selection.selected_microphones();
        let config = CaptureConfig {
            pids,
            output_path: path.clone(),
            sample_rate: self.sample_rate,
            microphones,
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

    /// Stop recording and load waveform
    pub(crate) fn stop_recording(&mut self, cx: &mut Context<Self>) {
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

    /// Poll for capture events from the recording session
    pub(crate) fn poll_capture_events(&mut self) {
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

    /// Reset session for new recording
    pub(crate) fn reset_session(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
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
}
