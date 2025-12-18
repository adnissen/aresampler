// Prevent console window from appearing on Windows for GUI application
#![windows_subsystem = "windows"]
use anyhow::Result;
use gpui::*;
use gpui_component::init as init_components;

mod app;
mod playback;
mod source_selection;
mod theme;
mod waveform;

fn main() -> Result<()> {
    // Initialize the GPUI application
    let app = Application::new();

    app.run(move |cx| {
        // Initialize gpui-component library
        init_components(cx);

        // Initialize theme
        theme::init(cx);

        // Configure window options - small centered window
        let window_options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(Bounds {
                origin: Point::default(),
                size: Size {
                    width: px(320.0),
                    height: px(550.0),
                },
            })),
            titlebar: Some(TitlebarOptions {
                title: None, // Empty title - we show our own header
                appears_transparent: true,
                traffic_light_position: Some(Point {
                    x: px(9.0),
                    y: px(9.0),
                }),
            }),
            ..Default::default()
        };

        // Open the main window
        cx.open_window(window_options, |window, cx| {
            cx.new(|cx| app::AppState::new(window, cx))
        })
        .expect("Failed to open window");

        cx.activate(true);
    });

    Ok(())
}
