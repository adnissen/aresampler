//! Waveform data, rendering, and trim controls

use super::AppState;
use gpui::prelude::FluentBuilder;
use gpui::*;
use gpui_component::{Theme, h_flex, v_flex};
use std::path::Path as StdPath;
use std::sync::Arc;

/// Represents the trim selection state with normalized positions (0.0 to 1.0)
#[derive(Clone)]
pub struct TrimSelection {
    /// Start position as a fraction of total duration (0.0 to 1.0)
    pub start: f32,
    /// End position as a fraction of total duration (0.0 to 1.0)
    pub end: f32,
}

impl Default for TrimSelection {
    fn default() -> Self {
        Self {
            start: 0.0,
            end: 1.0,
        }
    }
}

impl TrimSelection {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true if selection differs from default (full range)
    pub fn is_modified(&self) -> bool {
        self.start > 0.001 || self.end < 0.999
    }
}

/// Which handle is being dragged (if any)
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum DragHandle {
    Start,
    End,
}

/// Processed waveform data optimized for rendering.
/// Contains downsampled min/max pairs for efficient drawing at any width.
#[derive(Clone)]
pub struct WaveformData {
    /// Min/max sample pairs for each "bucket" of samples.
    /// Each pair represents the range of sample values in that time slice.
    pub peaks: Vec<(f32, f32)>,

    /// Total duration in seconds
    pub duration_secs: f64,
}

#[derive(Debug)]
pub enum WaveformError {
    Io(std::io::Error),
    Hound(hound::Error),
    UnsupportedFormat(String),
}

impl From<std::io::Error> for WaveformError {
    fn from(e: std::io::Error) -> Self {
        WaveformError::Io(e)
    }
}

impl From<hound::Error> for WaveformError {
    fn from(e: hound::Error) -> Self {
        WaveformError::Hound(e)
    }
}

impl std::fmt::Display for WaveformError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WaveformError::Io(e) => write!(f, "IO error: {}", e),
            WaveformError::Hound(e) => write!(f, "WAV error: {}", e),
            WaveformError::UnsupportedFormat(s) => write!(f, "Unsupported format: {}", s),
        }
    }
}

impl WaveformData {
    /// Load a WAV file and process it for display.
    ///
    /// # Arguments
    /// * `path` - Path to the WAV file
    /// * `target_buckets` - Target number of peak buckets (typically 2000-4000)
    ///
    /// This handles:
    /// - 32-bit float stereo WAV files
    /// - Mixing to mono (L+R average)
    /// - Downsampling via min/max peak detection
    pub fn load<P: AsRef<StdPath>>(path: P, target_buckets: usize) -> Result<Self, WaveformError> {
        let reader = hound::WavReader::open(path.as_ref())?;
        let spec = reader.spec();

        // Validate format expectations
        if spec.sample_format != hound::SampleFormat::Float {
            return Err(WaveformError::UnsupportedFormat(
                "Expected 32-bit float format".into(),
            ));
        }

        let channels = spec.channels as usize;
        let sample_rate = spec.sample_rate;
        let total_frames = reader.duration() as usize;

        if total_frames == 0 {
            return Ok(Self {
                peaks: vec![],
                duration_secs: 0.0,
            });
        }

        // Calculate samples per bucket for target resolution
        let samples_per_bucket = (total_frames / target_buckets).max(1);
        let actual_buckets = (total_frames + samples_per_bucket - 1) / samples_per_bucket;

        // Pre-allocate peaks vector
        let mut peaks: Vec<(f32, f32)> = Vec::with_capacity(actual_buckets);

        // Process samples
        let mut current_min: f32 = 0.0;
        let mut current_max: f32 = 0.0;
        let mut samples_in_bucket: usize = 0;

        // Buffer for reading interleaved samples
        let mut channel_samples: Vec<f32> = vec![0.0; channels];

        let mut samples_iter = reader.into_samples::<f32>();

        loop {
            // Read one frame (all channels)
            let mut got_samples = false;
            for ch in 0..channels {
                match samples_iter.next() {
                    Some(Ok(sample)) => {
                        channel_samples[ch] = sample;
                        got_samples = true;
                    }
                    Some(Err(_)) => {
                        channel_samples[ch] = 0.0;
                        got_samples = true;
                    }
                    None => break,
                }
            }

            if !got_samples {
                break;
            }

            // Mix to mono (average of all channels)
            let mono_sample: f32 = channel_samples.iter().sum::<f32>() / channels as f32;

            // Update min/max for current bucket
            if samples_in_bucket == 0 {
                current_min = mono_sample;
                current_max = mono_sample;
            } else {
                current_min = current_min.min(mono_sample);
                current_max = current_max.max(mono_sample);
            }

            samples_in_bucket += 1;

            // Complete bucket?
            if samples_in_bucket >= samples_per_bucket {
                peaks.push((current_min, current_max));
                samples_in_bucket = 0;
            }
        }

        // Handle remaining samples in last bucket
        if samples_in_bucket > 0 {
            peaks.push((current_min, current_max));
        }

        let duration_secs = total_frames as f64 / sample_rate as f64;

        Ok(Self {
            peaks,
            duration_secs,
        })
    }
}

/// The waveform view - renders a line waveform using GPUI canvas
pub struct WaveformView {
    data: Arc<WaveformData>,
    color: Hsla,
    line_width: f32,
    background: Hsla,
    vertical_padding: f32,
    trim_selection: Option<TrimSelection>,
    dimmed_color: Hsla,
    handle_color: Hsla,
    handle_width: f32,
}

impl WaveformView {
    pub fn new(data: Arc<WaveformData>) -> Self {
        // Default colors - will be overridden by theme
        Self {
            data,
            color: hsla(0.52, 0.82, 0.54, 1.0), // Default accent
            line_width: 1.5,
            background: hsla(0.0, 0.0, 0.04, 1.0), // Default dark background
            vertical_padding: 0.1,                 // 10% padding top and bottom
            trim_selection: None,
            dimmed_color: hsla(0.0, 0.0, 0.0, 0.6), // Semi-transparent black overlay
            handle_color: hsla(0.52, 0.82, 0.54, 1.0), // Default accent for handles
            handle_width: 4.0,
        }
    }

    pub fn with_trim_selection(mut self, selection: TrimSelection) -> Self {
        self.trim_selection = Some(selection);
        self
    }

    pub fn with_color(mut self, color: Hsla) -> Self {
        self.color = color;
        self
    }

    pub fn with_background(mut self, background: Hsla) -> Self {
        self.background = background;
        self
    }

    pub fn with_handle_color(mut self, color: Hsla) -> Self {
        self.handle_color = color;
        self
    }

    pub fn with_dimmed_color(mut self, color: Hsla) -> Self {
        self.dimmed_color = color;
        self
    }

    /// Render the waveform as an element using GPUI canvas
    pub fn render(self) -> impl IntoElement {
        let data = self.data;
        let color = self.color;
        let line_width = self.line_width;
        let background = self.background;
        let vertical_padding = self.vertical_padding;
        let trim_selection = self.trim_selection;
        let dimmed_color = self.dimmed_color;
        let handle_color = self.handle_color;
        let handle_width = self.handle_width;

        canvas(
            // Prepaint: calculate the path based on bounds
            move |bounds: Bounds<Pixels>, _window: &mut Window, _cx: &mut App| {
                let width: f32 = bounds.size.width.into();
                let height: f32 = bounds.size.height.into();

                if width <= 0.0 || height <= 0.0 || data.peaks.is_empty() {
                    return (
                        bounds,
                        None,
                        data.clone(),
                        background,
                        trim_selection.clone(),
                        dimmed_color,
                        handle_color,
                        handle_width,
                    );
                }

                // Calculate drawable area with padding
                let padding_px = height * vertical_padding;
                let drawable_height = height - (padding_px * 2.0);
                let center_y = height / 2.0;

                // Map bucket indices to x positions
                let num_buckets = data.peaks.len();
                let x_scale = width / num_buckets as f32;

                // Get origin as f32
                let origin_x: f32 = bounds.origin.x.into();
                let origin_y: f32 = bounds.origin.y.into();

                // Build the path - draw as a continuous line through the average values
                let mut builder = PathBuilder::stroke(px(line_width));

                // Start at first point
                let first_peak = &data.peaks[0];
                let first_val = (first_peak.0 + first_peak.1) / 2.0;
                let first_y = center_y - (first_val * drawable_height / 2.0);
                builder.move_to(point(px(origin_x), px(origin_y + first_y)));

                // Draw line to each subsequent point
                for (i, peak) in data.peaks.iter().enumerate().skip(1) {
                    let x = i as f32 * x_scale;
                    // Use the average of min/max for line position
                    let val = (peak.0 + peak.1) / 2.0;
                    let y = center_y - (val * drawable_height / 2.0);

                    builder.line_to(point(px(origin_x + x), px(origin_y + y)));
                }

                let path = builder.build().ok();
                (
                    bounds,
                    path,
                    data.clone(),
                    background,
                    trim_selection.clone(),
                    dimmed_color,
                    handle_color,
                    handle_width,
                )
            },
            // Paint: draw the background, path, trim overlays, and handles
            move |_bounds: Bounds<Pixels>,
                  (quad_bounds, path, _data, bg, trim_sel, dim_color, hdl_color, hdl_width): (
                Bounds<Pixels>,
                Option<Path<Pixels>>,
                Arc<WaveformData>,
                Hsla,
                Option<TrimSelection>,
                Hsla,
                Hsla,
                f32,
            ),
                  window: &mut Window,
                  _cx: &mut App| {
                // Draw background
                window.paint_quad(PaintQuad {
                    bounds: quad_bounds,
                    corner_radii: Corners::all(px(4.0)),
                    background: bg.into(),
                    border_widths: Edges::all(px(0.0)),
                    border_color: transparent_black(),
                    border_style: BorderStyle::default(),
                });

                // Draw the waveform path
                if let Some(path) = path {
                    window.paint_path(path, color);
                }

                // Draw trim overlays and handles if trim selection exists
                if let Some(trim) = trim_sel {
                    let origin_x: f32 = quad_bounds.origin.x.into();
                    let origin_y: f32 = quad_bounds.origin.y.into();
                    let width: f32 = quad_bounds.size.width.into();
                    let height: f32 = quad_bounds.size.height.into();

                    // Draw left dimmed region (0 to start)
                    if trim.start > 0.001 {
                        let dim_width = width * trim.start;
                        window.paint_quad(PaintQuad {
                            bounds: Bounds {
                                origin: point(px(origin_x), px(origin_y)),
                                size: Size {
                                    width: px(dim_width),
                                    height: px(height),
                                },
                            },
                            corner_radii: Corners {
                                top_left: px(4.0),
                                top_right: px(0.0),
                                bottom_left: px(4.0),
                                bottom_right: px(0.0),
                            },
                            background: dim_color.into(),
                            border_widths: Edges::all(px(0.0)),
                            border_color: transparent_black(),
                            border_style: BorderStyle::default(),
                        });
                    }

                    // Draw right dimmed region (end to 1)
                    if trim.end < 0.999 {
                        let end_x = width * trim.end;
                        let dim_width = width - end_x;
                        window.paint_quad(PaintQuad {
                            bounds: Bounds {
                                origin: point(px(origin_x + end_x), px(origin_y)),
                                size: Size {
                                    width: px(dim_width),
                                    height: px(height),
                                },
                            },
                            corner_radii: Corners {
                                top_left: px(0.0),
                                top_right: px(4.0),
                                bottom_left: px(0.0),
                                bottom_right: px(4.0),
                            },
                            background: dim_color.into(),
                            border_widths: Edges::all(px(0.0)),
                            border_color: transparent_black(),
                            border_style: BorderStyle::default(),
                        });
                    }

                    // Draw start handle (vertical bar)
                    let start_x = origin_x + (width * trim.start);
                    window.paint_quad(PaintQuad {
                        bounds: Bounds {
                            origin: point(px(start_x - hdl_width / 2.0), px(origin_y)),
                            size: Size {
                                width: px(hdl_width),
                                height: px(height),
                            },
                        },
                        corner_radii: Corners::all(px(2.0)),
                        background: hdl_color.into(),
                        border_widths: Edges::all(px(0.0)),
                        border_color: transparent_black(),
                        border_style: BorderStyle::default(),
                    });

                    // Draw end handle (vertical bar)
                    let end_x = origin_x + (width * trim.end);
                    window.paint_quad(PaintQuad {
                        bounds: Bounds {
                            origin: point(px(end_x - hdl_width / 2.0), px(origin_y)),
                            size: Size {
                                width: px(hdl_width),
                                height: px(height),
                            },
                        },
                        corner_radii: Corners::all(px(2.0)),
                        background: hdl_color.into(),
                        border_widths: Edges::all(px(0.0)),
                        border_color: transparent_black(),
                        border_style: BorderStyle::default(),
                    });
                }
            },
        )
        .size_full()
    }
}

/// Trim a WAV file to the specified normalized range and overwrite it
pub fn trim_wav_file(
    path: &StdPath,
    start_fraction: f32,
    end_fraction: f32,
) -> Result<(), WaveformError> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let total_frames = reader.duration() as usize;
    let channels = spec.channels as usize;

    // Validate format
    if spec.sample_format != hound::SampleFormat::Float {
        return Err(WaveformError::UnsupportedFormat(
            "Expected 32-bit float format".into(),
        ));
    }

    // Calculate frame ranges
    let start_frame = (total_frames as f32 * start_fraction) as usize;
    let end_frame = (total_frames as f32 * end_fraction) as usize;

    if end_frame <= start_frame {
        return Err(WaveformError::UnsupportedFormat(
            "Invalid trim range".into(),
        ));
    }

    // Read all samples into memory
    let samples: Vec<f32> = reader
        .into_samples::<f32>()
        .filter_map(|s| s.ok())
        .collect();

    // Extract the trimmed portion
    let start_sample = start_frame * channels;
    let end_sample = (end_frame * channels).min(samples.len());
    let trimmed_samples = &samples[start_sample..end_sample];

    // Write to a temp file first (safer approach for atomic operation)
    let temp_path = path.with_extension("wav.tmp");
    {
        let mut writer = hound::WavWriter::create(&temp_path, spec)?;
        for sample in trimmed_samples {
            writer.write_sample(*sample)?;
        }
        writer.finalize()?;
    }

    // Replace original with temp
    std::fs::rename(&temp_path, path)?;

    Ok(())
}

impl AppState {
    /// Render the waveform section with time display and controls
    pub(crate) fn render_waveform_section(
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
                        MouseButton::Left,
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
                        MouseButton::Left,
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
                            .with_dimmed_color(hsla(0.0, 0.0, 0.0, 0.5))
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
                                MouseButton::Left,
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
                                    MouseButton::Left,
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
                                MouseButton::Left,
                                cx.listener(|this, _, window, cx| {
                                    this.reset_session(window, cx);
                                }),
                            )
                            .child(div().text_color(theme.foreground).child("↺")),
                    ),
            )
    }
}
