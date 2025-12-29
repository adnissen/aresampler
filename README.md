`aresampler` is an application for Windows and Mac which records audio output from one or more processes or input devices and allows the user to trim the start and end time quickly. Each source is recorded into a separate channel in the resulting `.wav` file to make any further post processing easier.

Prebuilt releases are available for Windows and macOS (arm only).

<img height="400" alt="Screenshot 2025-12-04 204156" src="https://github.com/user-attachments/assets/8fe6492d-3482-490b-906b-876cf319c1f1" /> <img height="400" alt="Screenshot 2025-12-04 204328" src="https://github.com/user-attachments/assets/93164a49-82c2-40ec-8f71-73adacd475a1" />

To build and run the ui from source, execute: `cargo run`.

`aresampler`, with the exception of the theme files, is available under the AGPL license. The themes are from [https://github.com/longbridge/gpui-component](https://github.com/longbridge/gpui-component) and under the [https://github.com/longbridge/gpui-component/blob/main/LICENSE-APACHE](APACHE) license. See the individual `.json` files for further author information.

### Limitations
* On macOS, each application source will display the same volume level while recording. 
* * This is because Apple's `ScreenCaptureKit` API only provides a single stream of already-mixed audio, so it's not possible to tell which audio is coming from which source.
