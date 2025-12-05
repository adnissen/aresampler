`aresampler` is an application for Windows and Mac which records audio output from one or more processes and allows the user to trim the start and end time quickly.

<img height="400" alt="Screenshot 2025-12-04 204156" src="https://github.com/user-attachments/assets/8fe6492d-3482-490b-906b-876cf319c1f1" /> <img height="400" alt="Screenshot 2025-12-04 204328" src="https://github.com/user-attachments/assets/93164a49-82c2-40ec-8f71-73adacd475a1" />

There are two crates:
* `aresampler` - the gpui-based user interface
* `aresampler-core` - this is the set of modules which interact with the system apis and present a simpler interface for recording audio from a given set of PIDS

To build and run the ui from source, execute: `cargo run`

* `aresampler` is available under the AGPL license.
* `aresampler-core` is available under the Apache license. 
* See `LICENSE` in each crate directory for the full license text.
