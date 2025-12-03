`aresampler` is an application for Windows and Mac which records audio from one or more processes and allows the user to trim the start and end time quickly.

There are three crates:
* `aresampler-core` - this is the set of modules which interact with the system apis and present a simpler interface for recording audio from a given set of PIDS
* `aresampler-cli` - a command line implementation
* `aresampler-gui` - the gpui-based user interface

To build and run the ui from source, execute: `cargo run --bin aresampler-gui`

`aresampler-core` is available under the Apache license. `aresampler-cli` and `aresampler-gui` are available under the AGPL license. See `LICENSE` in each crate directory for the full license text.
