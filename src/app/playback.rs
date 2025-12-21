//! Audio playback controls and waveform mouse interaction (trim handles)

use super::waveform::{DragHandle, TrimSelection, WaveformData, WaveformError, trim_wav_file};
use super::AppState;
use gpui::{Context, Pixels, Point, Window};
use rodio::{OutputStream, OutputStreamHandle, Sink};
use std::ops::DerefMut;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug)]
pub enum PlaybackError {
    #[allow(dead_code)]
    Stream(rodio::StreamError),
    #[allow(dead_code)]
    Play(rodio::PlayError),
}

impl From<rodio::StreamError> for PlaybackError {
    fn from(e: rodio::StreamError) -> Self {
        PlaybackError::Stream(e)
    }
}

impl From<rodio::PlayError> for PlaybackError {
    fn from(e: rodio::PlayError) -> Self {
        PlaybackError::Play(e)
    }
}

pub struct AudioPlayer {
    _stream: OutputStream,
    _handle: OutputStreamHandle,
    sink: Sink,
}

impl AudioPlayer {
    pub fn new() -> Result<Self, PlaybackError> {
        let (stream, handle) = OutputStream::try_default()?;
        let sink = Sink::try_new(&handle)?;
        Ok(Self {
            _stream: stream,
            _handle: handle,
            sink,
        })
    }

    /// Play audio samples with the given sample rate and channel count
    pub fn play_samples(&self, samples: Vec<f32>, sample_rate: u32, channels: u16) {
        self.sink.stop();
        let source = rodio::buffer::SamplesBuffer::new(channels, sample_rate, samples);
        self.sink.append(source);
        self.sink.play();
    }

    /// Stop playback
    pub fn stop(&self) {
        self.sink.stop();
    }

    /// Returns true if playback has finished
    pub fn is_empty(&self) -> bool {
        self.sink.empty()
    }
}

/// Load samples from a WAV file for a specific region
/// If the WAV has more than 2 channels, it will be mixed down to stereo for playback
pub fn load_samples_for_region(
    path: &Path,
    start_fraction: f32,
    end_fraction: f32,
) -> Result<(Vec<f32>, u32, u16), String> {
    let reader = hound::WavReader::open(path).map_err(|e| format!("Failed to open WAV: {}", e))?;
    let spec = reader.spec();

    if spec.sample_format != hound::SampleFormat::Float {
        return Err("Expected 32-bit float format".into());
    }

    let total_frames = reader.duration() as usize;
    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate;

    // Calculate frame ranges
    let start_frame = (total_frames as f32 * start_fraction) as usize;
    let end_frame = (total_frames as f32 * end_fraction) as usize;

    // Read all samples
    let all_samples: Vec<f32> = reader
        .into_samples::<f32>()
        .filter_map(|s| s.ok())
        .collect();

    // Extract the region
    let start_sample = start_frame * channels;
    let end_sample = (end_frame * channels).min(all_samples.len());
    let region_samples = &all_samples[start_sample..end_sample];

    // If 4-channel (app L/R + mic L/R), mix down to stereo for playback
    if channels == 4 {
        let num_frames = region_samples.len() / 4;
        let mut stereo_samples = Vec::with_capacity(num_frames * 2);

        for frame in 0..num_frames {
            let base = frame * 4;
            let app_l = region_samples.get(base).copied().unwrap_or(0.0);
            let app_r = region_samples.get(base + 1).copied().unwrap_or(0.0);
            let mic_l = region_samples.get(base + 2).copied().unwrap_or(0.0);
            let mic_r = region_samples.get(base + 3).copied().unwrap_or(0.0);

            // Mix: sum app and mic, clamp to prevent clipping
            let left = (app_l + mic_l).clamp(-1.0, 1.0);
            let right = (app_r + mic_r).clamp(-1.0, 1.0);

            stereo_samples.push(left);
            stereo_samples.push(right);
        }

        Ok((stereo_samples, sample_rate, 2))
    } else {
        Ok((region_samples.to_vec(), sample_rate, spec.channels))
    }
}

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
                Err(_) => Err(WaveformError::Io(std::io::Error::new(
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
