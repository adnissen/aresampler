//! Cross-platform process utilities

use crate::types::ProcessInfo;
use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};

/// Check if a process with the given PID exists
pub fn process_exists(pid: u32) -> bool {
    let refresh = RefreshKind::new().with_processes(ProcessRefreshKind::new());
    let sys = System::new_with_specifics(refresh);
    sys.process(Pid::from_u32(pid)).is_some()
}

/// Get detailed process information
pub fn get_process_info(pid: u32) -> Option<ProcessInfo> {
    let refresh = RefreshKind::new().with_processes(ProcessRefreshKind::everything());
    let sys = System::new_with_specifics(refresh);

    let process = sys.process(Pid::from_u32(pid))?;

    Some(ProcessInfo {
        pid,
        name: process.name().to_string_lossy().to_string(),
        exe_path: process.exe().map(|p| p.to_string_lossy().to_string()),
    })
}

/// Get the parent process ID for a given process
pub fn get_parent_pid(pid: u32) -> Option<u32> {
    let refresh = RefreshKind::new().with_processes(ProcessRefreshKind::everything());
    let sys = System::new_with_specifics(refresh);

    let process = sys.process(Pid::from_u32(pid))?;
    process.parent().map(|p| p.as_u32())
}
