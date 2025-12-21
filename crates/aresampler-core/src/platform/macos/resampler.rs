//! Audio resampler for handling sample rate mismatches
//!
//! Uses rubato for high-quality sinc interpolation resampling.

use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

/// Resampler that handles sample rate conversion for audio streams.
/// Lazily initializes when a sample rate mismatch is detected.
pub struct AudioResampler {
    resampler: Option<SincFixedIn<f32>>,
    current_input_rate: Option<u32>,
    output_rate: u32,
    channels: usize,
    /// Buffer for resampler output
    output_buffer: Vec<Vec<f32>>,
}

impl AudioResampler {
    /// Create a new resampler targeting the specified output sample rate.
    pub fn new(output_rate: u32, channels: usize) -> Self {
        Self {
            resampler: None,
            current_input_rate: None,
            output_rate,
            channels,
            output_buffer: vec![Vec::new(); channels],
        }
    }

    /// Process audio samples, resampling if the input rate differs from output rate.
    ///
    /// Returns resampled samples if resampling was performed, or None if rates match.
    /// Input samples are expected in planar format: one Vec<f32> per channel.
    pub fn process(&mut self, input_rate: u32, samples: &[Vec<f32>]) -> Option<Vec<Vec<f32>>> {
        // If rates match, no resampling needed
        if input_rate == self.output_rate {
            return None;
        }

        // Initialize or reinitialize resampler if input rate changed
        if self.current_input_rate != Some(input_rate) {
            self.init_resampler(input_rate);
        }

        let resampler = self.resampler.as_mut()?;

        // Process the samples
        match resampler.process(samples, None) {
            Ok(output) => Some(output),
            Err(e) => {
                eprintln!("Resampling error: {:?}", e);
                None
            }
        }
    }

    /// Initialize the resampler for a specific input rate
    fn init_resampler(&mut self, input_rate: u32) {
        let resample_ratio = self.output_rate as f64 / input_rate as f64;

        // High quality sinc interpolation parameters
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };

        // Create the resampler
        // chunk_size is the expected input size - use a reasonable default
        // that will be expanded as needed
        let chunk_size = 1024;

        match SincFixedIn::new(
            resample_ratio,
            2.0, // max relative ratio (allows for some rate variation)
            params,
            chunk_size,
            self.channels,
        ) {
            Ok(r) => {
                self.resampler = Some(r);
                self.current_input_rate = Some(input_rate);
                eprintln!(
                    "Audio resampler initialized: {} Hz -> {} Hz (ratio: {:.4})",
                    input_rate, self.output_rate, resample_ratio
                );
            }
            Err(e) => {
                eprintln!("Failed to create resampler: {:?}", e);
                self.resampler = None;
                self.current_input_rate = None;
            }
        }
    }

    /// Get the target output sample rate
    pub fn output_rate(&self) -> u32 {
        self.output_rate
    }
}
