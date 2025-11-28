/// A fixed-capacity ring buffer for interleaved audio samples.
///
/// Stores interleaved f32 samples (L, R, L, R, ...) and automatically
/// overwrites the oldest samples when the buffer is full.
#[derive(Debug)]
pub struct AudioRingBuffer {
    buffer: Vec<f32>,
    capacity: usize,
    write_pos: usize,
    filled: usize,
    sample_rate: u32,
    channels: u16,
}

impl AudioRingBuffer {
    /// Create a new ring buffer with capacity for `duration_secs` of audio.
    ///
    /// # Arguments
    /// * `duration_secs` - Maximum duration of audio to store
    /// * `sample_rate` - Sample rate in Hz (e.g., 48000)
    /// * `channels` - Number of channels (e.g., 2 for stereo)
    pub fn new(duration_secs: f32, sample_rate: u32, channels: u16) -> Self {
        let samples_per_second = sample_rate as usize * channels as usize;
        let capacity = (duration_secs * samples_per_second as f32) as usize;

        Self {
            buffer: vec![0.0; capacity],
            capacity,
            write_pos: 0,
            filled: 0,
            sample_rate,
            channels,
        }
    }

    /// Push interleaved samples into the buffer.
    ///
    /// If the buffer is full, the oldest samples are overwritten.
    pub fn push(&mut self, samples: &[f32]) {
        if self.capacity == 0 {
            return;
        }

        for &sample in samples {
            self.buffer[self.write_pos] = sample;
            self.write_pos = (self.write_pos + 1) % self.capacity;
            if self.filled < self.capacity {
                self.filled += 1;
            }
        }
    }

    /// Drain all buffered samples in chronological order.
    ///
    /// Returns a Vec containing all samples from oldest to newest.
    /// The buffer is cleared after draining.
    pub fn drain(&mut self) -> Vec<f32> {
        if self.filled == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(self.filled);

        if self.filled < self.capacity {
            // Buffer hasn't wrapped yet - data is contiguous from start
            result.extend_from_slice(&self.buffer[..self.filled]);
        } else {
            // Buffer has wrapped - oldest data starts at write_pos
            result.extend_from_slice(&self.buffer[self.write_pos..]);
            result.extend_from_slice(&self.buffer[..self.write_pos]);
        }

        self.clear();
        result
    }

    /// Get the current fill level in seconds.
    pub fn duration_secs(&self) -> f32 {
        let samples_per_second = self.sample_rate as usize * self.channels as usize;
        if samples_per_second == 0 {
            return 0.0;
        }
        self.filled as f32 / samples_per_second as f32
    }

    /// Get the maximum capacity in seconds.
    pub fn capacity_secs(&self) -> f32 {
        let samples_per_second = self.sample_rate as usize * self.channels as usize;
        if samples_per_second == 0 {
            return 0.0;
        }
        self.capacity as f32 / samples_per_second as f32
    }

    /// Clear all buffered samples.
    pub fn clear(&mut self) {
        self.write_pos = 0;
        self.filled = 0;
    }

    /// Check if the buffer has any data.
    pub fn is_empty(&self) -> bool {
        self.filled == 0
    }

    /// Get the number of samples currently stored.
    pub fn len(&self) -> usize {
        self.filled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_buffer() {
        let buf = AudioRingBuffer::new(1.0, 48000, 2);
        assert_eq!(buf.capacity, 96000); // 1 sec * 48000 Hz * 2 channels
        assert_eq!(buf.filled, 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_push_and_drain_partial() {
        let mut buf = AudioRingBuffer::new(1.0, 48000, 2);

        // Push some samples (less than capacity)
        let samples: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        buf.push(&samples);

        assert_eq!(buf.len(), 1000);
        assert!(!buf.is_empty());

        let drained = buf.drain();
        assert_eq!(drained.len(), 1000);
        assert_eq!(drained, samples);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_push_and_drain_full_wrap() {
        // Small buffer for easy testing
        let mut buf = AudioRingBuffer::new(0.001, 1000, 1); // 1 sample capacity
        assert_eq!(buf.capacity, 1);

        buf.push(&[1.0]);
        buf.push(&[2.0]); // This should overwrite the first

        let drained = buf.drain();
        assert_eq!(drained, vec![2.0]);
    }

    #[test]
    fn test_wrap_around() {
        // Buffer for 10 samples
        let mut buf = AudioRingBuffer::new(0.01, 1000, 1);
        assert_eq!(buf.capacity, 10);

        // Push 15 samples - should wrap and keep last 10
        let samples: Vec<f32> = (0..15).map(|i| i as f32).collect();
        buf.push(&samples);

        assert_eq!(buf.len(), 10);
        let drained = buf.drain();
        // Should have samples 5-14
        let expected: Vec<f32> = (5..15).map(|i| i as f32).collect();
        assert_eq!(drained, expected);
    }

    #[test]
    fn test_duration_secs() {
        let mut buf = AudioRingBuffer::new(10.0, 48000, 2);

        // Empty buffer
        assert_eq!(buf.duration_secs(), 0.0);

        // Push 1 second of audio (48000 * 2 samples)
        let samples = vec![0.0; 96000];
        buf.push(&samples);
        assert!((buf.duration_secs() - 1.0).abs() < 0.001);

        // Capacity should be 10 seconds
        assert!((buf.capacity_secs() - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_clear() {
        let mut buf = AudioRingBuffer::new(1.0, 48000, 2);
        buf.push(&[1.0, 2.0, 3.0, 4.0]);

        assert!(!buf.is_empty());
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_zero_duration() {
        let buf = AudioRingBuffer::new(0.0, 48000, 2);
        assert_eq!(buf.capacity, 0);
        assert!(buf.is_empty());
    }
}
