// Prevent console window from appearing on Windows for GUI application
#![windows_subsystem = "windows"]
use anyhow::Result;
use gpui::*;
use gpui_component::init as init_components;
#[cfg(target_os = "macos")]
use gpui_component::Theme;

mod app;
mod assets;
mod config;
mod core;
mod source_selection;
mod theme;

// Action to quit the application
actions!([Quit]);

/// Action to change the application theme
#[derive(Action, Clone, PartialEq)]
#[action(no_json)]
pub struct SwitchTheme(pub SharedString);

fn main() -> Result<()> {
    // Initialize the GPUI application with embedded assets
    let app = Application::new().with_assets(assets::Assets);

    app.run(move |cx| {
        // Initialize gpui-component library
        init_components(cx);

        // Load config and initialize theme with saved preference
        let app_config = config::Config::load();
        theme::init_with_theme_name(app_config.theme_name.as_deref(), cx);

        // Register the quit action handler
        cx.on_action(|_: &Quit, cx| {
            cx.quit();
        });

        // Register the theme change action handler
        cx.on_action(|action: &SwitchTheme, cx| {
            let registry = theme::ThemeRegistry::new();
            if let Some(theme_variant) = registry.find_theme(&action.0) {
                theme::apply_theme(&theme_variant.config, cx);
                // Refresh all windows to apply the theme
                cx.refresh_windows();

                // Save the theme selection to config
                let mut app_config = config::Config::load();
                app_config.theme_name = Some(action.0.to_string());
                let _ = app_config.save();

                // Update the menu to show the new checkmark
                #[cfg(target_os = "macos")]
                {
                    setup_macos_menu(cx);
                }
            }
        });

        // Set up macOS menu bar (only on macOS)
        #[cfg(target_os = "macos")]
        {
            setup_macos_menu(cx);
        }

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

/// Set up the macOS application menu bar
#[cfg(target_os = "macos")]
fn setup_macos_menu(cx: &mut App) {
    use theme::ThemeRegistry;

    let registry = ThemeRegistry::new();
    let current_theme_name = Theme::global(cx).theme_name().to_string();

    // Build theme submenu items with checkmark on current theme
    let theme_items: Vec<MenuItem> = registry
        .themes
        .iter()
        .map(|theme_variant| {
            let is_current = theme_variant.name == current_theme_name;
            MenuItem::action(
                theme_variant.name.clone(),
                SwitchTheme(theme_variant.name.clone().into()),
            )
            .checked(is_current)
        })
        .collect();

    cx.set_menus(vec![
        // Application menu (macOS standard)
        Menu {
            name: "aresampler".into(),
            items: vec![MenuItem::action("Quit aresampler", Quit)],
        },
        // Theme menu with all available themes
        Menu {
            name: "Theme".into(),
            items: theme_items,
        },
    ]);
}
