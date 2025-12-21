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

    /// Resize the buffer to a new duration.
    ///
    /// - Shrinking: keeps the newest samples, drops oldest
    /// - Expanding: existing samples become the oldest in the new buffer
    pub fn resize(&mut self, new_duration_secs: f32) {
        let samples_per_second = self.sample_rate as usize * self.channels as usize;
        let new_capacity = (new_duration_secs * samples_per_second as f32) as usize;

        if new_capacity == self.capacity {
            return; // No change needed
        }

        if new_capacity == 0 {
            self.buffer = Vec::new();
            self.capacity = 0;
            self.write_pos = 0;
            self.filled = 0;
            return;
        }

        // Extract current samples in chronological order (oldest to newest)
        let current_samples = self.drain();

        // Create new buffer
        self.buffer = vec![0.0; new_capacity];
        self.capacity = new_capacity;

        if current_samples.is_empty() {
            self.write_pos = 0;
            self.filled = 0;
            return;
        }

        if new_capacity >= current_samples.len() {
            // Expanding: copy all samples, they become the "oldest"
            self.buffer[..current_samples.len()].copy_from_slice(&current_samples);
            self.write_pos = current_samples.len() % new_capacity;
            self.filled = current_samples.len();
        } else {
            // Shrinking: keep only the newest samples (skip oldest)
            let skip = current_samples.len() - new_capacity;
            self.buffer.copy_from_slice(&current_samples[skip..]);
            self.write_pos = 0; // Buffer is now full, write_pos wraps to 0
            self.filled = new_capacity;
        }
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

    #[test]
    fn test_resize_shrink_full_buffer() {
        // Buffer for 10 samples, shrink to 5
        let mut buf = AudioRingBuffer::new(0.01, 1000, 1);
        assert_eq!(buf.capacity, 10);

        // Fill with samples 0-9
        let samples: Vec<f32> = (0..10).map(|i| i as f32).collect();
        buf.push(&samples);
        assert_eq!(buf.len(), 10);

        // Shrink to 5 samples - should keep newest (5-9)
        buf.resize(0.005);
        assert_eq!(buf.capacity, 5);
        assert_eq!(buf.len(), 5);

        let drained = buf.drain();
        let expected: Vec<f32> = (5..10).map(|i| i as f32).collect();
        assert_eq!(drained, expected);
    }

    #[test]
    fn test_resize_shrink_partial_buffer() {
        // Buffer for 10 samples, only 3 filled, shrink to 5
        let mut buf = AudioRingBuffer::new(0.01, 1000, 1);
        buf.push(&[1.0, 2.0, 3.0]);

        buf.resize(0.005);
        assert_eq!(buf.capacity, 5);
        assert_eq!(buf.len(), 3);

        let drained = buf.drain();
        assert_eq!(drained, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_resize_expand_full_buffer() {
        // Buffer for 5 samples, expand to 10
        let mut buf = AudioRingBuffer::new(0.005, 1000, 1);
        assert_eq!(buf.capacity, 5);

        // Fill with samples 0-4
        let samples: Vec<f32> = (0..5).map(|i| i as f32).collect();
        buf.push(&samples);

        // Expand to 10 - existing samples become oldest
        buf.resize(0.01);
        assert_eq!(buf.capacity, 10);
        assert_eq!(buf.len(), 5);

        // Push 3 more samples
        buf.push(&[10.0, 11.0, 12.0]);
        assert_eq!(buf.len(), 8);

        let drained = buf.drain();
        // Should have original 5 (oldest) followed by 3 new
        assert_eq!(drained, vec![0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_resize_expand_partial_buffer() {
        // Buffer for 5 samples, 3 filled, expand to 10
        let mut buf = AudioRingBuffer::new(0.005, 1000, 1);
        buf.push(&[1.0, 2.0, 3.0]);

        buf.resize(0.01);
        assert_eq!(buf.capacity, 10);
        assert_eq!(buf.len(), 3);

        let drained = buf.drain();
        assert_eq!(drained, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_resize_to_zero() {
        let mut buf = AudioRingBuffer::new(0.01, 1000, 1);
        buf.push(&[1.0, 2.0, 3.0]);

        buf.resize(0.0);
        assert_eq!(buf.capacity, 0);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_resize_same_size() {
        let mut buf = AudioRingBuffer::new(0.01, 1000, 1);
        buf.push(&[1.0, 2.0, 3.0]);

        buf.resize(0.01);
        assert_eq!(buf.capacity, 10);
        assert_eq!(buf.len(), 3);

        let drained = buf.drain();
        assert_eq!(drained, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_resize_with_wrapped_buffer() {
        // Buffer for 5 samples, push 8 to cause wrap, then resize
        let mut buf = AudioRingBuffer::new(0.005, 1000, 1);
        assert_eq!(buf.capacity, 5);

        // Push 8 samples - buffer wraps, keeps 3-7
        let samples: Vec<f32> = (0..8).map(|i| i as f32).collect();
        buf.push(&samples);
        assert_eq!(buf.len(), 5);

        // Shrink to 3 - should keep newest (5, 6, 7)
        buf.resize(0.003);
        assert_eq!(buf.capacity, 3);
        assert_eq!(buf.len(), 3);

        let drained = buf.drain();
        assert_eq!(drained, vec![5.0, 6.0, 7.0]);
    }
}
