`aresampler` is an application for Windows and Mac which records audio output from one or more processes or input devices and allows the user to trim the start and end time quickly. Each source is recorded into a separate channel in the resulting `.wav` file to make any further post processing easier.

Prebuilt releases are available for Windows and macOS (arm only).

<img width="200" height="979" alt="Screenshot 2025-12-29 105900" src="https://github.com/user-attachments/assets/30ad2abd-ed63-4227-833b-d8d5560991b3" /><img width="200" height="979" alt="Screenshot 2025-12-29 105907" src="https://github.com/user-attachments/assets/256212c9-e4b1-4635-85a0-c1bda0d7f12d" /><img width="200" height="979" alt="Screenshot 2025-12-29 105920" src="https://github.com/user-attachments/assets/99285b71-bf0b-41f0-be1a-bc481c348c39" />


To build and run the ui from source, execute: `cargo run`.

`aresampler`, with the exception of the theme files, is available under the AGPL license. The themes are from [https://github.com/longbridge/gpui-component](https://github.com/longbridge/gpui-component) and under the [APACHE](https://github.com/longbridge/gpui-component/blob/main/LICENSE-APACHE) license. See the individual `.json` files for further author information.

### Limitations
* On macOS, each application source will display the same volume level while recording. 
* * This is because Apple's `ScreenCaptureKit` API only provides a single stream of already-mixed audio, so it's not possible to tell which audio is coming from which source.
