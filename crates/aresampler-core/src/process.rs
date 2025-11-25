use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};

#[derive(Clone, Debug)]
pub struct ProcessInfo {
    pub pid: u32,
    pub name: String,
    pub exe_path: Option<String>,
}

/// Check if a process with the given PID exists
pub fn process_exists(pid: u32) -> bool {
    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;

    #[link(name = "kernel32")]
    extern "system" {
        fn OpenProcess(
            dwDesiredAccess: u32,
            bInheritHandle: i32,
            dwProcessId: u32,
        ) -> *mut std::ffi::c_void;
        fn CloseHandle(hObject: *mut std::ffi::c_void) -> i32;
    }

    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            false
        } else {
            CloseHandle(handle);
            true
        }
    }
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
