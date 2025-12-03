`aresampler-core` captures audio output from one or more processes and writes the result to a WAV file.

## Platform Implementations

### Windows

Uses Windows Audio Session API (`WASAPI`) application loopback capture:

- `AudioClient::new_application_loopback_client()` - per-process audio capture
- `IAudioCaptureClient` - audio buffer reads
- Event-driven capture with 200ms buffer

App enumeration via Win32:
- `EnumWindows()` / `GetWindowThreadProcessId()` - visible window → PID mapping

### macOS

Uses `ScreenCaptureKit`:

- `SCShareableContent` - app/window enumeration
- `SCStream` with `SCContentFilter` - per-app audio capture
- `CMSampleBuffer` - audio data extraction (planar → interleaved conversion)

Requires `Screen Recording` permission.

## Usage

```rust
use aresampler_core::*;

// Initialize (required on Windows, no-op on macOS)
initialize_audio()?;

// Check/request permission (macOS requires Screen Recording permission)
if !is_capture_available()? {
    request_capture_permission()?;
}

// List capturable applications
let sessions: Vec<AudioSessionInfo> = enumerate_audio_sessions()?;

// API for immediately starting recording
let config = CaptureConfig {
    pids: vec![target_pid],
    output_path: PathBuf::from("output.wav"),
    sample_rate: 48000,
    channels: 2,
    bits_per_sample: 32,
};
let mut capture = CaptureSession::new(config);
let events: Receiver<CaptureEvent> = capture.start()?;
// ... handle events ...
capture.stop()?;

// API for "Monitoring" mode (ring buffer "pre-roll")
// this keeps saving and overwriting for x amount of time and 
// then adds it to the start of the recording when `start_recording` is called
let monitor_config = MonitorConfig {
    pids: vec![target_pid],
    sample_rate: 48000,
    channels: 2,
    pre_roll_duration_secs: 10.0,
};
let mut capture = CaptureSession::new_empty();
let events = capture.start_monitoring(monitor_config)?;
// Later, start recording (includes buffered pre-roll audio):
capture.start_recording(PathBuf::from("output.wav"))?;
```

### Types

```rust
pub struct AudioSessionInfo {
    pub pid: u32,
    pub name: String,
    pub bundle_id: Option<String>, //mac only
    pub icon_png: Option<Vec<u8>>,
}

pub struct CaptureStats {
    pub duration_secs: f64,
    pub total_frames: u64,
    pub file_size_bytes: u64,
    pub is_recording: bool,
    pub is_monitoring: bool,
    pub pre_roll_buffer_secs: f32,
    pub left_rms_db: f32,
    pub right_rms_db: f32,
}

pub enum CaptureEvent {
    Started { buffer_size: usize },
    StatsUpdate(CaptureStats),
    Stopped,
    Error(String),
    MonitoringStarted,
    RecordingStarted { pre_roll_secs: f32 },
}
```
