use anyhow::Result;
use gpui::*;
use gpui_component::init as init_components;

mod app;

fn main() -> Result<()> {
    // Initialize the GPUI application
    let app = Application::new();

    app.run(move |cx| {
        // Initialize gpui-component library
        init_components(cx);

        // Configure window options - small centered window
        let window_options = WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(Bounds {
                origin: Point::default(),
                size: Size {
                    width: px(450.0),
                    height: px(350.0),
                },
            })),
            titlebar: Some(TitlebarOptions {
                title: Some("Aresampler".into()),
                ..Default::default()
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
