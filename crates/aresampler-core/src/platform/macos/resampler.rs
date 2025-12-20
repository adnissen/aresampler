//! Audio resampler for handling sample rate mismatches
//!
//! Uses rubato for high-quality sinc interpolation resampling.

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

/// Resampler that handles sample rate conversion for audio streams.
/// Lazily initializes when a sample rate mismatch is detected.
/// Automatically adapts to the actual number of channels in the input.
pub struct AudioResampler {
    resampler: Option<SincFixedIn<f32>>,
    current_input_rate: Option<u32>,
    current_channels: Option<usize>,
    output_rate: u32,
}

impl AudioResampler {
    /// Create a new resampler targeting the specified output sample rate.
    pub fn new(output_rate: u32, _channels: usize) -> Self {
        Self {
            resampler: None,
            current_input_rate: None,
            current_channels: None,
            output_rate,
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

        let num_channels = samples.len();
        if num_channels == 0 {
            return None;
        }

        // Initialize or reinitialize resampler if input rate or channel count changed
        let needs_reinit = self.current_input_rate != Some(input_rate)
            || self.current_channels != Some(num_channels);

        if needs_reinit {
            self.init_resampler(input_rate, num_channels);
        }

        let resampler = self.resampler.as_mut()?;

        // Process the samples using process_partial to handle variable buffer sizes
        match resampler.process_partial(Some(samples), None) {
            Ok(output) => Some(output),
            Err(e) => {
                eprintln!("Resampling error: {:?}", e);
                None
            }
        }
    }

    /// Initialize the resampler for a specific input rate and channel count
    fn init_resampler(&mut self, input_rate: u32, channels: usize) {
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
            channels,
        ) {
            Ok(r) => {
                self.resampler = Some(r);
                self.current_input_rate = Some(input_rate);
                self.current_channels = Some(channels);
                eprintln!(
                    "Audio resampler initialized: {} Hz -> {} Hz, {} channels (ratio: {:.4})",
                    input_rate, self.output_rate, channels, resample_ratio
                );
            }
            Err(e) => {
                eprintln!("Failed to create resampler: {:?}", e);
                self.resampler = None;
                self.current_input_rate = None;
                self.current_channels = None;
            }
        }
    }
}
