`aresampler` is an application for Windows and Mac which records audio output from one or more processes and allows the user to trim the start and end time quickly. Prebuilt releases are available for Windows and macOS (arm only).

<img height="400" alt="Screenshot 2025-12-04 204156" src="https://github.com/user-attachments/assets/8fe6492d-3482-490b-906b-876cf319c1f1" /> <img height="400" alt="Screenshot 2025-12-04 204328" src="https://github.com/user-attachments/assets/93164a49-82c2-40ec-8f71-73adacd475a1" />

To build and run the ui from source, execute: `cargo run`.

`aresampler` is available under the AGPL license.

### Limitations
* On macOS, each source will display the same volume level while recording. 
* * This is because Apple's `ScreenCaptureKit` API only provides a single stream of already-mixed audio, so it's not possible to tell which audio is coming from which source.
