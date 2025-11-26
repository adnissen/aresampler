//! Windows audio session enumeration using WASAPI

use crate::types::AudioSessionInfo;
use anyhow::Result;
use std::collections::HashSet;
use std::thread;
use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};
use wasapi::{DeviceCollection, Direction};

/// Enumerate all processes with active WASAPI audio sessions
///
/// This function spawns a separate thread because WASAPI requires COM in MTA mode,
/// but GUI frameworks like GPUI initialize COM in STA mode.
pub fn enumerate_audio_sessions() -> Result<Vec<AudioSessionInfo>> {
    let handle = thread::spawn(enumerate_audio_sessions_impl);
    handle
        .join()
        .map_err(|_| anyhow::anyhow!("Thread panicked"))?
        .map_err(|e| e.into())
}

/// Internal implementation that must run in a thread with MTA COM initialization
fn enumerate_audio_sessions_impl() -> Result<Vec<AudioSessionInfo>> {
    // Initialize COM in MTA mode for WASAPI
    let hr = wasapi::initialize_mta();
    if hr.is_err() {
        return Err(anyhow::anyhow!(
            "Failed to initialize COM: HRESULT {:#x}",
            hr.0
        ));
    }

    let mut sessions = Vec::new();
    let mut seen_pids: HashSet<u32> = HashSet::new();

    // Get system info for process lookups
    let refresh = RefreshKind::new().with_processes(ProcessRefreshKind::everything());
    let sys = System::new_with_specifics(refresh);

    // Enumerate render devices (audio output - what applications play)
    let devices = DeviceCollection::new(&Direction::Render)?;

    for device in &devices {
        let dev = match device {
            Ok(d) => d,
            Err(_) => continue,
        };

        // Get the session manager for this device
        let manager = match dev.get_iaudiosessionmanager() {
            Ok(m) => m,
            Err(_) => continue,
        };

        // Get the session enumerator
        let enumerator = match manager.get_audiosessionenumerator() {
            Ok(e) => e,
            Err(_) => continue,
        };

        // Get session count
        let count = match enumerator.get_count() {
            Ok(c) => c,
            Err(_) => continue,
        };

        // Iterate through all sessions
        for i in 0..count {
            let control = match enumerator.get_session(i) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let pid = match control.get_process_id() {
                Ok(p) => p,
                Err(_) => continue,
            };

            // Skip system sounds (PID 0)
            if pid == 0 {
                continue;
            }

            // Resolve to parent process
            let resolved_pid = resolve_to_parent(&sys, pid);

            // Skip already seen PIDs
            if seen_pids.contains(&resolved_pid) {
                continue;
            }
            seen_pids.insert(resolved_pid);

            // Get process name
            if let Some(process) = sys.process(Pid::from_u32(resolved_pid)) {
                sessions.push(AudioSessionInfo {
                    pid: resolved_pid,
                    name: process.name().to_string_lossy().to_string(),
                    bundle_id: None, // Windows doesn't have bundle IDs
                });
            }
        }
    }

    // Sort by process name for consistent display
    sessions.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));

    Ok(sessions)
}

/// Resolve a PID to its parent process if appropriate
fn resolve_to_parent(sys: &System, pid: u32) -> u32 {
    let Some(process) = sys.process(Pid::from_u32(pid)) else {
        return pid;
    };

    // Check if this process has a parent
    let Some(parent_pid) = process.parent() else {
        return pid;
    };

    // Get parent process info
    let Some(parent) = sys.process(parent_pid) else {
        return pid;
    };

    let parent_name = parent.name().to_string_lossy().to_lowercase();

    // Don't resolve to system processes
    if is_system_process(&parent_name) {
        return pid;
    }

    // Return the parent PID
    parent_pid.as_u32()
}

fn is_system_process(name: &str) -> bool {
    matches!(
        name,
        "explorer.exe"
            | "svchost.exe"
            | "services.exe"
            | "system"
            | "csrss.exe"
            | "wininit.exe"
            | "winlogon.exe"
            | "dwm.exe"
            | "sihost.exe"
            | "taskhostw.exe"
    )
}
