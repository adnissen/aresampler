//! Source selection component for choosing which application to record audio from.
//!
//! This module provides the `ProcessItem` type for representing audio sources and
//! the `SourceSelectionState` struct for managing source selection state.

use crate::core::{
    AudioSessionInfo, InputDevice, enumerate_audio_sessions, enumerate_input_devices,
    get_app_icon_png,
};
use gpui::{
    AnyElement, App, AppContext, Context, Entity, ImageSource, IntoElement, ParentElement,
    RenderImage, SharedString, Styled, Window, div, img, prelude::FluentBuilder, px,
};
use gpui_component::{
    Theme, h_flex,
    select::{SearchableVec, SelectItem, SelectState},
};
use std::collections::HashMap;
use std::sync::Arc;


/// Represents an audio source (application/process) that can be recorded from.
///
/// This is a wrapper around `AudioSessionInfo` that implements `SelectItem`
/// for use with the gpui-component Select dropdown.
#[derive(Clone)]
pub struct ProcessItem {
    /// Process ID
    pub pid: u32,
    /// Application name
    pub name: String,
    /// Pre-rendered icon for GPUI display
    pub icon: Option<Arc<RenderImage>>,
}

impl ProcessItem {
    /// Create a ProcessItem from AudioSessionInfo, using the icon cache for efficiency.
    /// Icons are fetched on-demand and cached to avoid repeated slow OS calls.
    pub fn from_audio_session(
        info: AudioSessionInfo,
        icon_cache: &mut HashMap<String, Option<Arc<RenderImage>>>,
    ) -> Self {
        // Use bundle_id or exe_path as cache key, fall back to name
        let cache_key = info
            .bundle_id
            .clone()
            .or_else(|| info.exe_path.clone())
            .unwrap_or_else(|| info.name.clone());

        // Check cache first (includes cached None for apps without icons)
        let icon = if let Some(cached_icon) = icon_cache.get(&cache_key) {
            cached_icon.clone()
        } else {
            // Not in cache - fetch the icon (slow operation, but only once per app)
            #[cfg(target_os = "macos")]
            let png_bytes = info
                .bundle_id
                .as_ref()
                .and_then(|bid| get_app_icon_png(bid));

            #[cfg(target_os = "windows")]
            let png_bytes = info
                .exe_path
                .as_ref()
                .and_then(|path| get_app_icon_png(std::path::Path::new(path)));

            #[cfg(not(any(target_os = "macos", target_os = "windows")))]
            let png_bytes: Option<Vec<u8>> = None;

            // Decode PNG to RenderImage
            let new_icon = png_bytes.and_then(|bytes| {
                let img = image::load_from_memory(&bytes).ok()?;
                let mut rgba = img.to_rgba8();

                // Convert RGBA to BGRA (GPUI expects BGRA format)
                for pixel in rgba.chunks_exact_mut(4) {
                    pixel.swap(0, 2); // Swap R and B channels
                }

                let frame = image::Frame::new(rgba);
                Some(Arc::new(RenderImage::new(vec![frame])))
            });

            // Cache the result (even if None, to avoid re-fetching)
            icon_cache.insert(cache_key, new_icon.clone());
            new_icon
        };

        Self {
            pid: info.pid,
            name: info.name,
            icon,
        }
    }
}

/// Represents an audio input device (microphone) that can be selected for recording.
#[derive(Clone)]
pub struct InputDeviceItem {
    /// Unique device identifier
    pub id: String,
    /// Device name
    pub name: String,
    /// Whether this is the system default microphone
    pub is_default: bool,
}

impl InputDeviceItem {
    /// Create an InputDeviceItem from InputDevice
    pub fn from_input_device(device: InputDevice) -> Self {
        Self {
            id: device.id,
            name: device.name,
            is_default: device.is_default,
        }
    }
}

/// Value type for unified source selection (can be app PID or microphone ID)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SourceItemValue {
    App(u32),
    Microphone(String),
}

/// Unified source item that can be either an application or a microphone
#[derive(Clone)]
pub enum SourceItem {
    App(ProcessItem),
    Microphone(InputDeviceItem),
}

impl SourceItem {
    /// Get the name of this source
    pub fn name(&self) -> &str {
        match self {
            SourceItem::App(p) => &p.name,
            SourceItem::Microphone(m) => &m.name,
        }
    }

    /// Check if this is a microphone
    pub fn is_microphone(&self) -> bool {
        matches!(self, SourceItem::Microphone(_))
    }

    /// Get as ProcessItem if this is an app
    pub fn as_app(&self) -> Option<&ProcessItem> {
        match self {
            SourceItem::App(p) => Some(p),
            _ => None,
        }
    }

    /// Get as InputDeviceItem if this is a microphone
    pub fn as_microphone(&self) -> Option<&InputDeviceItem> {
        match self {
            SourceItem::Microphone(m) => Some(m),
            _ => None,
        }
    }
}

impl SelectItem for SourceItem {
    type Value = SourceItemValue;

    fn value(&self) -> &Self::Value {
        // We need to store the value, so we'll use a leaked reference
        // This is a workaround since SelectItem expects &Self::Value
        match self {
            SourceItem::App(p) => {
                // Use a static approach - store the value inline
                Box::leak(Box::new(SourceItemValue::App(p.pid)))
            }
            SourceItem::Microphone(m) => {
                Box::leak(Box::new(SourceItemValue::Microphone(m.id.clone())))
            }
        }
    }

    fn title(&self) -> SharedString {
        match self {
            SourceItem::App(p) => p.name.clone().into(),
            SourceItem::Microphone(m) => {
                if m.is_default {
                    format!("🎤 {} (Default)", m.name).into()
                } else {
                    format!("🎤 {}", m.name).into()
                }
            }
        }
    }

    fn display_title(&self) -> Option<AnyElement> {
        match self {
            SourceItem::App(p) => p.display_title(),
            SourceItem::Microphone(m) => Some(
                h_flex()
                    .gap_2()
                    .items_center()
                    .child(
                        div()
                            .size(px(16.0))
                            .flex_shrink_0()
                            .rounded(px(4.0))
                            .bg(gpui::rgb(0x4a9eff))
                            .flex()
                            .items_center()
                            .justify_center()
                            .text_xs()
                            .child("🎤"),
                    )
                    .child(div().overflow_hidden().text_ellipsis().child(if m.is_default {
                        format!("{} (Default)", m.name)
                    } else {
                        m.name.clone()
                    }))
                    .into_any_element(),
            ),
        }
    }

    fn render(&self, _window: &mut Window, _cx: &mut App) -> impl IntoElement {
        match self {
            SourceItem::App(p) => h_flex()
                .gap_2()
                .items_center()
                .when_some(p.icon.clone(), |this, icon| {
                    this.child(
                        img(ImageSource::Render(icon))
                            .size(px(16.0))
                            .flex_shrink_0(),
                    )
                })
                .when(p.icon.is_none(), |this| {
                    this.child(div().size(px(16.0)).flex_shrink_0())
                })
                .child(div().overflow_hidden().text_ellipsis().child(p.name.clone()))
                .into_any_element(),
            SourceItem::Microphone(m) => h_flex()
                .gap_2()
                .items_center()
                .child(
                    div()
                        .size(px(16.0))
                        .flex_shrink_0()
                        .rounded(px(4.0))
                        .bg(gpui::rgb(0x4a9eff))
                        .flex()
                        .items_center()
                        .justify_center()
                        .text_xs()
                        .child("🎤"),
                )
                .child(div().overflow_hidden().text_ellipsis().child(if m.is_default {
                    format!("{} (Default)", m.name)
                } else {
                    m.name.clone()
                }))
                .into_any_element(),
        }
    }
}

impl SelectItem for ProcessItem {
    type Value = u32;

    fn value(&self) -> &Self::Value {
        &self.pid
    }

    fn title(&self) -> SharedString {
        format!("{}", self.name).into()
    }

    fn display_title(&self) -> Option<AnyElement> {
        Some(
            h_flex()
                .gap_2()
                .items_center()
                .when_some(self.icon.clone(), |this, icon| {
                    this.child(
                        img(ImageSource::Render(icon))
                            .size(px(16.0))
                            .flex_shrink_0(),
                    )
                })
                .child(div().overflow_hidden().text_ellipsis().child(self.title()))
                .into_any_element(),
        )
    }

    fn render(&self, _window: &mut Window, _cx: &mut App) -> impl IntoElement {
        h_flex()
            .gap_2()
            .items_center()
            .when_some(self.icon.clone(), |this, icon| {
                this.child(
                    img(ImageSource::Render(icon))
                        .size(px(16.0))
                        .flex_shrink_0(),
                )
            })
            .when(self.icon.is_none(), |this| {
                // Empty placeholder to maintain alignment when no icon
                this.child(div().size(px(16.0)).flex_shrink_0())
            })
            .child(div().overflow_hidden().text_ellipsis().child(self.title()))
    }
}

/// A single source selection entry with its own dropdown state and selected source.
pub struct SourceEntry {
    /// Select dropdown state for this source (contains both apps and microphones)
    pub select_state: Entity<SelectState<SearchableVec<SourceItem>>>,
    /// Currently selected source (can be app or microphone)
    pub selected_source: Option<SourceItem>,
}

/// State for the source selection component, supporting multiple sources.
pub struct SourceSelectionState {
    /// List of available audio sources (applications)
    pub processes: Vec<ProcessItem>,
    /// List of available audio input devices (microphones)
    pub input_devices: Vec<InputDeviceItem>,
    /// Multiple source entries (each with its own dropdown and selection)
    pub sources: Vec<SourceEntry>,
    /// Icon cache: maps bundle_id/exe_path to rendered icon (or None if icon unavailable)
    icon_cache: HashMap<String, Option<Arc<RenderImage>>>,
}

impl SourceSelectionState {
    /// Create a new source selection state with one initial source entry.
    pub fn new<V: 'static>(has_permission: bool, window: &mut Window, cx: &mut Context<V>) -> Self {
        // Initialize icon cache and enumerate processes
        let mut icon_cache = HashMap::new();
        let processes: Vec<ProcessItem> = if has_permission {
            enumerate_audio_sessions()
                .unwrap_or_default()
                .into_iter()
                .map(|info| ProcessItem::from_audio_session(info, &mut icon_cache))
                .collect()
        } else {
            Vec::new()
        };

        // Enumerate input devices (microphones)
        let input_devices: Vec<InputDeviceItem> = enumerate_input_devices()
            .unwrap_or_default()
            .into_iter()
            .map(InputDeviceItem::from_input_device)
            .collect();

        // Build unified source items: microphones first, then apps
        let source_items = Self::build_source_items(&input_devices, &processes, &[]);

        // Create initial source entry
        let searchable = SearchableVec::new(source_items);
        let select_state = cx.new(|cx| SelectState::new(searchable, None, window, cx));
        let initial_source = SourceEntry {
            select_state,
            selected_source: None,
        };

        Self {
            processes,
            input_devices,
            sources: vec![initial_source],
            icon_cache,
        }
    }

    /// Build unified source items list with microphones first, then apps
    fn build_source_items(
        input_devices: &[InputDeviceItem],
        processes: &[ProcessItem],
        exclude_values: &[SourceItemValue],
    ) -> Vec<SourceItem> {
        let mut items = Vec::new();

        // Add microphones first (excluding already-selected ones)
        for mic in input_devices {
            let value = SourceItemValue::Microphone(mic.id.clone());
            if !exclude_values.contains(&value) {
                items.push(SourceItem::Microphone(mic.clone()));
            }
        }

        // Add apps (excluding already-selected ones)
        for process in processes {
            let value = SourceItemValue::App(process.pid);
            if !exclude_values.contains(&value) {
                items.push(SourceItem::App(process.clone()));
            }
        }

        items
    }

    /// Get the list of all selected app PIDs (excludes microphones).
    pub fn selected_pids(&self) -> Vec<u32> {
        self.sources
            .iter()
            .filter_map(|s| {
                s.selected_source
                    .as_ref()
                    .and_then(|src| src.as_app())
                    .map(|p| p.pid)
            })
            .collect()
    }

    /// Get the selected microphone device ID (if any source is a microphone).
    pub fn selected_microphone_id(&self) -> Option<String> {
        self.sources.iter().find_map(|s| {
            s.selected_source
                .as_ref()
                .and_then(|src| src.as_microphone())
                .map(|m| m.id.clone())
        })
    }

    /// Get the values of all selected sources (for filtering).
    fn selected_values(&self) -> Vec<SourceItemValue> {
        self.sources
            .iter()
            .filter_map(|s| {
                s.selected_source.as_ref().map(|src| match src {
                    SourceItem::App(p) => SourceItemValue::App(p.pid),
                    SourceItem::Microphone(m) => SourceItemValue::Microphone(m.id.clone()),
                })
            })
            .collect()
    }

    /// Add a new source entry and return its index.
    pub fn add_source<V: 'static>(&mut self, window: &mut Window, cx: &mut Context<V>) -> usize {
        // Create filtered list excluding already-selected sources
        let exclude = self.selected_values();
        let source_items = Self::build_source_items(&self.input_devices, &self.processes, &exclude);

        let searchable = SearchableVec::new(source_items);
        let select_state = cx.new(|cx| SelectState::new(searchable, None, window, cx));
        let new_source = SourceEntry {
            select_state,
            selected_source: None,
        };

        self.sources.push(new_source);
        self.sources.len() - 1
    }

    /// Remove a source entry by index.
    pub fn remove_source(&mut self, index: usize) {
        if index > 0 && index < self.sources.len() {
            self.sources.remove(index);
        }
    }

    /// Refresh the list of available processes and input devices, update all dropdowns.
    pub fn refresh_processes<V: 'static>(&mut self, window: &mut Window, cx: &mut Context<V>) {
        // Re-enumerate processes
        self.processes = enumerate_audio_sessions()
            .unwrap_or_default()
            .into_iter()
            .map(|info| ProcessItem::from_audio_session(info, &mut self.icon_cache))
            .collect();

        // Re-enumerate input devices
        self.input_devices = enumerate_input_devices()
            .unwrap_or_default()
            .into_iter()
            .map(InputDeviceItem::from_input_device)
            .collect();

        // Update each source's dropdown
        self.update_all_dropdowns(window, cx);

        // Clear all selections
        for source in &mut self.sources {
            source.selected_source = None;
        }
    }

    /// Refresh a specific source's dropdown, filtering out already-selected sources.
    pub fn refresh_source_dropdown<V: 'static>(
        &mut self,
        index: usize,
        window: &mut Window,
        cx: &mut Context<V>,
    ) {
        // Re-enumerate to discover new apps/devices
        self.processes = enumerate_audio_sessions()
            .unwrap_or_default()
            .into_iter()
            .map(|info| ProcessItem::from_audio_session(info, &mut self.icon_cache))
            .collect();

        self.input_devices = enumerate_input_devices()
            .unwrap_or_default()
            .into_iter()
            .map(InputDeviceItem::from_input_device)
            .collect();

        // Get values selected by other entries (not this one)
        let exclude: Vec<SourceItemValue> = self
            .sources
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if i != index {
                    s.selected_source.as_ref().map(|src| match src {
                        SourceItem::App(p) => SourceItemValue::App(p.pid),
                        SourceItem::Microphone(m) => SourceItemValue::Microphone(m.id.clone()),
                    })
                } else {
                    None
                }
            })
            .collect();

        let source_items = Self::build_source_items(&self.input_devices, &self.processes, &exclude);

        if let Some(source) = self.sources.get(index) {
            let searchable = SearchableVec::new(source_items);
            source.select_state.update(cx, |state, cx| {
                state.set_items(searchable, window, cx);
            });
        }
    }

    /// Update all dropdowns to filter out already-selected sources.
    pub fn update_all_dropdowns<V: 'static>(&mut self, window: &mut Window, cx: &mut Context<V>) {
        for i in 0..self.sources.len() {
            self.refresh_source_dropdown(i, window, cx);
        }
    }

    /// Find a source item by value.
    pub fn find_source(&self, value: &SourceItemValue) -> Option<SourceItem> {
        match value {
            SourceItemValue::App(pid) => self
                .processes
                .iter()
                .find(|p| p.pid == *pid)
                .map(|p| SourceItem::App(p.clone())),
            SourceItemValue::Microphone(id) => self
                .input_devices
                .iter()
                .find(|m| m.id == *id)
                .map(|m| SourceItem::Microphone(m.clone())),
        }
    }

    /// Set the selected source for a specific entry.
    pub fn set_selected(&mut self, index: usize, source: SourceItem) {
        if let Some(entry) = self.sources.get_mut(index) {
            entry.selected_source = Some(source);
        }
    }

    /// Check if any source has a selection.
    pub fn has_any_selection(&self) -> bool {
        self.sources.iter().any(|s| s.selected_source.is_some())
    }

    /// Get the first selected source.
    pub fn first_selected_source(&self) -> Option<&SourceItem> {
        self.sources
            .first()
            .and_then(|s| s.selected_source.as_ref())
    }
}

/// Render a placeholder icon for empty source selection.
pub fn render_placeholder_icon(cx: &App) -> impl IntoElement {
    let theme = Theme::global(cx);

    div()
        .size(px(28.0))
        .rounded(px(6.0))
        .bg(theme.muted)
        .border_1()
        .border_color(theme.border)
        .flex()
        .items_center()
        .justify_center()
        .child(
            div()
                .size(px(14.0))
                .rounded(px(3.0))
                .border_1()
                .border_color(theme.muted_foreground),
        )
}
