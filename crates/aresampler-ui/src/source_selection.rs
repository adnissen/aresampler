//! Source selection component for choosing which application to record audio from.
//!
//! This module provides the `ProcessItem` type for representing audio sources and
//! the `SourceSelectionState` struct for managing source selection state.

use aresampler_core::{AudioSessionInfo, enumerate_audio_sessions, get_app_icon_png};
use gpui::{
    AnyElement, App, AppContext, Context, Entity, ImageSource, IntoElement, ParentElement,
    RenderImage, SharedString, Styled, Window, div, img, prelude::FluentBuilder, px,
};
use gpui_component::{
    h_flex,
    select::{SearchableVec, SelectItem, SelectState},
};
use std::collections::HashMap;
use std::sync::Arc;

use crate::app::colors;

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

/// A single source selection entry with its own dropdown state and selected process.
pub struct SourceEntry {
    /// Select dropdown state for this source
    pub select_state: Entity<SelectState<SearchableVec<ProcessItem>>>,
    /// Currently selected process for this source
    pub selected_process: Option<ProcessItem>,
}

/// State for the source selection component, supporting multiple sources.
pub struct SourceSelectionState {
    /// List of available audio sources
    pub processes: Vec<ProcessItem>,
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

        // Create initial source entry
        let searchable = SearchableVec::new(processes.clone());
        let select_state = cx.new(|cx| SelectState::new(searchable, None, window, cx));
        let initial_source = SourceEntry {
            select_state,
            selected_process: None,
        };

        Self {
            processes,
            sources: vec![initial_source],
            icon_cache,
        }
    }

    /// Get the list of all selected PIDs.
    pub fn selected_pids(&self) -> Vec<u32> {
        self.sources
            .iter()
            .filter_map(|s| s.selected_process.as_ref().map(|p| p.pid))
            .collect()
    }

    /// Add a new source entry and return its index.
    pub fn add_source<V: 'static>(&mut self, window: &mut Window, cx: &mut Context<V>) -> usize {
        // Create filtered list excluding already-selected processes
        let selected_pids = self.selected_pids();
        let filtered: Vec<ProcessItem> = self
            .processes
            .iter()
            .filter(|p| !selected_pids.contains(&p.pid))
            .cloned()
            .collect();

        let searchable = SearchableVec::new(filtered);
        let select_state = cx.new(|cx| SelectState::new(searchable, None, window, cx));
        let new_source = SourceEntry {
            select_state,
            selected_process: None,
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

    /// Refresh the list of available processes and update all source dropdowns.
    pub fn refresh_processes<V: 'static>(&mut self, window: &mut Window, cx: &mut Context<V>) {
        // Use the icon cache to avoid re-fetching icons for known processes
        self.processes = enumerate_audio_sessions()
            .unwrap_or_default()
            .into_iter()
            .map(|info| ProcessItem::from_audio_session(info, &mut self.icon_cache))
            .collect();

        // Update each source's dropdown with filtered items
        self.update_all_dropdowns(window, cx);

        // Clear all selections
        for source in &mut self.sources {
            source.selected_process = None;
        }
    }

    /// Refresh a specific source's dropdown, filtering out already-selected processes.
    pub fn refresh_source_dropdown<V: 'static>(
        &mut self,
        index: usize,
        window: &mut Window,
        cx: &mut Context<V>,
    ) {
        // Re-enumerate audio sessions to discover new apps
        self.processes = enumerate_audio_sessions()
            .unwrap_or_default()
            .into_iter()
            .map(|info| ProcessItem::from_audio_session(info, &mut self.icon_cache))
            .collect();

        let selected_pids: Vec<u32> = self
            .sources
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                if i != index {
                    s.selected_process.as_ref().map(|p| p.pid)
                } else {
                    None
                }
            })
            .collect();

        let filtered: Vec<ProcessItem> = self
            .processes
            .iter()
            .filter(|p| !selected_pids.contains(&p.pid))
            .cloned()
            .collect();

        if let Some(source) = self.sources.get(index) {
            let searchable = SearchableVec::new(filtered);
            source.select_state.update(cx, |state, cx| {
                state.set_items(searchable, window, cx);
            });
        }
    }

    /// Update all dropdowns to filter out already-selected processes.
    pub fn update_all_dropdowns<V: 'static>(&mut self, window: &mut Window, cx: &mut Context<V>) {
        for i in 0..self.sources.len() {
            self.refresh_source_dropdown(i, window, cx);
        }
    }

    /// Find a process by PID.
    pub fn find_process(&self, pid: u32) -> Option<&ProcessItem> {
        self.processes.iter().find(|p| p.pid == pid)
    }

    /// Set the selected process for a specific source.
    pub fn set_selected(&mut self, index: usize, process: ProcessItem) {
        if let Some(source) = self.sources.get_mut(index) {
            source.selected_process = Some(process);
        }
    }

    /// Clear the selection for a specific source.
    pub fn clear_selection<V: 'static>(
        &mut self,
        index: usize,
        window: &mut Window,
        cx: &mut Context<V>,
    ) {
        if let Some(source) = self.sources.get_mut(index) {
            source.selected_process = None;
            source.select_state.update(cx, |state, cx| {
                state.set_selected_index(None, window, cx);
            });
        }
    }

    /// Clear all selections.
    pub fn clear_all_selections<V: 'static>(&mut self, window: &mut Window, cx: &mut Context<V>) {
        for i in 0..self.sources.len() {
            self.clear_selection(i, window, cx);
        }
    }

    /// Check if any source has a selection.
    pub fn has_any_selection(&self) -> bool {
        self.sources.iter().any(|s| s.selected_process.is_some())
    }

    /// Get the first selected process (for backwards compatibility).
    pub fn first_selected_process(&self) -> Option<&ProcessItem> {
        self.sources
            .first()
            .and_then(|s| s.selected_process.as_ref())
    }
}

/// Render a placeholder icon for empty source selection.
pub fn render_placeholder_icon() -> impl IntoElement {
    div()
        .size(px(28.0))
        .rounded(px(6.0))
        .bg(colors::bg_tertiary())
        .border_1()
        .border_color(colors::border())
        .flex()
        .items_center()
        .justify_center()
        .child(
            div()
                .size(px(14.0))
                .rounded(px(3.0))
                .border_1()
                .border_color(colors::text_muted()),
        )
}
