use rodio::{OutputStream, OutputStreamHandle, Sink};
use std::path::Path;

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
    let region_samples = all_samples[start_sample..end_sample].to_vec();

    Ok((region_samples, sample_rate, spec.channels))
}
