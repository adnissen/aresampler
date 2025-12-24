//! Audio resampler for handling sample rate mismatches
//!
//! Uses rubato for high-quality sinc interpolation resampling.

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

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

        // Process the samples using process_partial for variable-length input
        match resampler.process_partial(Some(samples), None) {
            Ok(output) => Some(output),
            Err(e) => {
                eprintln!("Resampling error: {:?}", e);
                None
            }
        }
    }

    /// Process interleaved audio samples, resampling if needed.
    ///
    /// This is a convenience method that handles de-interleaving, resampling,
    /// and re-interleaving. If rates match, returns the input unchanged.
    ///
    /// # Arguments
    /// * `input_rate` - The sample rate of the input audio
    /// * `interleaved` - Interleaved samples (L0, R0, L1, R1, ...)
    ///
    /// # Returns
    /// Interleaved resampled samples, or the original samples if rates match.
    pub fn process_interleaved(&mut self, input_rate: u32, interleaved: &[f32]) -> Vec<f32> {
        // If rates match, return input unchanged
        if input_rate == self.output_rate {
            return interleaved.to_vec();
        }

        // De-interleave into planar format
        let num_frames = interleaved.len() / self.channels;
        let mut planar: Vec<Vec<f32>> = vec![Vec::with_capacity(num_frames); self.channels];

        for (i, &sample) in interleaved.iter().enumerate() {
            let channel = i % self.channels;
            planar[channel].push(sample);
        }

        // Resample
        if let Some(resampled) = self.process(input_rate, &planar) {
            // Re-interleave
            let output_frames = resampled.first().map(|v| v.len()).unwrap_or(0);
            let mut output = Vec::with_capacity(output_frames * self.channels);

            for frame_idx in 0..output_frames {
                for ch in 0..self.channels {
                    output.push(resampled[ch].get(frame_idx).copied().unwrap_or(0.0));
                }
            }
            output
        } else {
            // Resampling failed, return original
            interleaved.to_vec()
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
        // process_partial handles variable-length input
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
