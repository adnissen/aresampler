`aresampler` is an application for Windows and Mac which records audio from one or more processes and allows the user to trim the start and end time quickly.

There are three crates:
* `aresampler` - the gpui-based user interface
* `aresampler-core` - this is the set of modules which interact with the system apis and present a simpler interface for recording audio from a given set of PIDS

To build and run the ui from source, execute: `cargo run`

* `aresampler` is available under the AGPL license.
* `aresampler-core` is available under the Apache license. 
* See `LICENSE` in each crate directory for the full license text.
