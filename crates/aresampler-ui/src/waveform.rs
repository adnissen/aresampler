use gpui::*;
use std::path::Path as StdPath;
use std::sync::Arc;

/// Processed waveform data optimized for rendering.
/// Contains downsampled min/max pairs for efficient drawing at any width.
#[derive(Clone)]
pub struct WaveformData {
    /// Min/max sample pairs for each "bucket" of samples.
    /// Each pair represents the range of sample values in that time slice.
    pub peaks: Vec<(f32, f32)>,

    /// Original sample rate (for time calculations)
    pub sample_rate: u32,

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
                sample_rate,
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
            sample_rate,
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
}

impl WaveformView {
    pub fn new(data: Arc<WaveformData>) -> Self {
        Self {
            data,
            color: hsla(0.55, 0.7, 0.5, 1.0), // Cyan-ish blue
            line_width: 1.5,
            background: hsla(0.0, 0.0, 0.12, 1.0), // Dark gray
            vertical_padding: 0.1,                  // 10% padding top and bottom
        }
    }

    /// Render the waveform as an element using GPUI canvas
    pub fn render(self) -> impl IntoElement {
        let data = self.data;
        let color = self.color;
        let line_width = self.line_width;
        let background = self.background;
        let vertical_padding = self.vertical_padding;

        canvas(
            // Prepaint: calculate the path based on bounds
            move |bounds: Bounds<Pixels>, _window: &mut Window, _cx: &mut App| {
                let width: f32 = bounds.size.width.into();
                let height: f32 = bounds.size.height.into();

                if width <= 0.0 || height <= 0.0 || data.peaks.is_empty() {
                    return (bounds, None, data.clone(), background);
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
                (bounds, path, data.clone(), background)
            },
            // Paint: draw the background and path
            move |_bounds: Bounds<Pixels>,
                  (quad_bounds, path, _data, bg): (
                Bounds<Pixels>,
                Option<Path<Pixels>>,
                Arc<WaveformData>,
                Hsla,
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
            },
        )
        .size_full()
    }
}
