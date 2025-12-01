//! Source selection component for choosing which application to record audio from.
//!
//! This module provides the `ProcessItem` type for representing audio sources and
//! the `SourceSelectionState` struct for managing source selection state.

use aresampler_core::{enumerate_audio_sessions, AudioSessionInfo};
use gpui::{
    div, img, prelude::FluentBuilder, px, AnyElement, App, AppContext, Context, Entity,
    ImageSource, IntoElement, ParentElement, RenderImage, SharedString, Styled, Window,
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
    /// If the icon is not in the cache, it will be decoded and added.
    pub fn from_audio_session(
        info: AudioSessionInfo,
        icon_cache: &mut HashMap<String, Arc<RenderImage>>,
    ) -> Self {
        // Use bundle_id as cache key, fall back to name if no bundle_id
        let cache_key = info.bundle_id.clone().unwrap_or_else(|| info.name.clone());

        // Check cache first
        let icon = if let Some(cached_icon) = icon_cache.get(&cache_key) {
            Some(cached_icon.clone())
        } else {
            // Not in cache - decode and cache it
            let new_icon = info.icon_png.and_then(|png_bytes| {
                let img = image::load_from_memory(&png_bytes).ok()?;
                let mut rgba = img.to_rgba8();

                // Convert RGBA to BGRA (GPUI expects BGRA format)
                for pixel in rgba.chunks_exact_mut(4) {
                    pixel.swap(0, 2); // Swap R and B channels
                }

                let frame = image::Frame::new(rgba);
                Some(Arc::new(RenderImage::new(vec![frame])))
            });

            // Cache the icon if we got one
            if let Some(ref icon) = new_icon {
                icon_cache.insert(cache_key, icon.clone());
            }

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

/// State for the source selection component.
pub struct SourceSelectionState {
    /// List of available audio sources
    pub processes: Vec<ProcessItem>,
    /// Select dropdown state
    pub select_state: Entity<SelectState<SearchableVec<ProcessItem>>>,
    /// Currently selected process
    pub selected_process: Option<ProcessItem>,
    /// Icon cache: maps bundle_id (or name) to rendered icon
    icon_cache: HashMap<String, Arc<RenderImage>>,
}

impl SourceSelectionState {
    /// Create a new source selection state.
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
        let searchable = SearchableVec::new(processes.clone());

        // Create select state
        let select_state = cx.new(|cx| SelectState::new(searchable, None, window, cx));

        Self {
            processes,
            select_state,
            selected_process: None,
            icon_cache,
        }
    }

    /// Refresh the list of available processes.
    pub fn refresh_processes<V: 'static>(&mut self, window: &mut Window, cx: &mut Context<V>) {
        // Use the icon cache to avoid re-decoding icons for known processes
        self.processes = enumerate_audio_sessions()
            .unwrap_or_default()
            .into_iter()
            .map(|info| ProcessItem::from_audio_session(info, &mut self.icon_cache))
            .collect();
        let searchable = SearchableVec::new(self.processes.clone());

        // Update select state with new items
        self.select_state.update(cx, |state, _cx| {
            state.set_items(searchable, window, _cx);
        });

        self.selected_process = None;
    }

    /// Find a process by PID.
    pub fn find_process(&self, pid: u32) -> Option<&ProcessItem> {
        self.processes.iter().find(|p| p.pid == pid)
    }

    /// Set the selected process.
    pub fn set_selected(&mut self, process: ProcessItem) {
        self.selected_process = Some(process);
    }

    /// Clear the selection.
    pub fn clear_selection<V: 'static>(&mut self, window: &mut Window, cx: &mut Context<V>) {
        self.selected_process = None;
        self.select_state.update(cx, |state, cx| {
            state.set_selected_index(None, window, cx);
        });
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
