//! Windows application enumeration using GUI window enumeration
//!
//! This approach enumerates all visible GUI applications, similar to how
//! macOS ScreenCaptureKit returns all capturable applications.

use crate::types::AudioSessionInfo;
use anyhow::Result;
use std::collections::HashSet;
use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};
use windows::Win32::Foundation::{BOOL, HWND, LPARAM};
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GetWindowTextLengthW, GetWindowTextW, GetWindowThreadProcessId, IsWindowVisible,
};

/// Enumerate all GUI applications that could potentially produce audio
///
/// Returns a list of running applications with visible windows.
/// This matches the macOS behavior where all capturable apps are shown,
/// not just those currently playing audio.
pub fn enumerate_audio_sessions() -> Result<Vec<AudioSessionInfo>> {
    let mut sessions = Vec::new();
    let mut seen_pids: HashSet<u32> = HashSet::new();

    // Get system info for process lookups
    let refresh = RefreshKind::new().with_processes(ProcessRefreshKind::everything());
    let sys = System::new_with_specifics(refresh);

    // Collect PIDs from visible windows
    let mut window_pids: Vec<u32> = Vec::new();

    unsafe {
        // EnumWindows calls our callback for each top-level window
        let _ = EnumWindows(
            Some(enum_windows_callback),
            LPARAM(&mut window_pids as *mut Vec<u32> as isize),
        );
    }

    // Process each unique PID
    for pid in window_pids {
        // Skip system PIDs
        if pid == 0 || pid == 4 {
            continue;
        }

        // Skip already seen PIDs
        if seen_pids.contains(&pid) {
            continue;
        }
        seen_pids.insert(pid);

        // Get process info
        if let Some(process) = sys.process(Pid::from_u32(pid)) {
            let name = process.name().to_string_lossy().to_string();

            // Skip known system processes that shouldn't appear in the list
            let name_lower = name.to_lowercase();
            if is_system_process(&name_lower) {
                continue;
            }

            let exe_path = process.exe().map(|p| p.to_string_lossy().to_string());

            sessions.push(AudioSessionInfo {
                pid,
                name,
                bundle_id: None, // Windows doesn't have bundle IDs
                exe_path,
                icon_png: None, // Icons fetched on-demand by UI layer
            });
        }
    }

    // Sort by process name for consistent display
    sessions.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));

    Ok(sessions)
}

/// Callback function for EnumWindows
///
/// Called for each top-level window. Filters for visible windows with titles
/// and collects their process IDs.
unsafe extern "system" fn enum_windows_callback(hwnd: HWND, lparam: LPARAM) -> BOOL {
    let pids = &mut *(lparam.0 as *mut Vec<u32>);

    // Skip invisible windows
    if !IsWindowVisible(hwnd).as_bool() {
        return BOOL(1); // Continue enumeration
    }

    // Skip windows without a title (likely background/system windows)
    let title_len = GetWindowTextLengthW(hwnd);
    if title_len == 0 {
        return BOOL(1); // Continue enumeration
    }

    // Get the window title to verify it's a real window
    let mut title_buf = vec![0u16; (title_len + 1) as usize];
    let actual_len = GetWindowTextW(hwnd, &mut title_buf);
    if actual_len == 0 {
        return BOOL(1); // Continue enumeration
    }

    // Get the process ID for this window
    let mut pid: u32 = 0;
    GetWindowThreadProcessId(hwnd, Some(&mut pid));

    if pid != 0 {
        pids.push(pid);
    }

    BOOL(1) // Continue enumeration
}

/// Check if a process name is a known system process that shouldn't be shown
fn is_system_process(name: &str) -> bool {
    matches!(
        name,
        "applicationframehost.exe"  // UWP app host
            | "shellexperiencehost.exe"  // Windows shell
            | "searchhost.exe"  // Windows search
            | "startmenuexperiencehost.exe"  // Start menu
            | "textinputhost.exe"  // Touch keyboard
            | "lockapp.exe"  // Lock screen
            | "systemsettings.exe" // Settings (internal)
    )
}
