//! Scoped AMDGPU userptr intent; all other libdrm calls pass through unchanged.
use std::{cell::Cell, ffi::c_void, sync::OnceLock};

#[derive(Clone, Copy, Default)]
struct Intent {
    address: u64,
    length: u64,
    flags: u32,
}
thread_local! { static INTENT: Cell<Intent> = const { Cell::new(Intent { address: 0, length: 0, flags: 0 }) }; }

pub(super) fn with_import<R>(
    address: usize,
    length: usize,
    writable: bool,
    run: impl FnOnce() -> R,
) -> R {
    struct Restore(Intent);
    impl Drop for Restore {
        fn drop(&mut self) {
            let _ = INTENT.try_with(|cell| cell.set(self.0));
        }
    }
    let previous = INTENT.with(|cell| {
        cell.replace(Intent {
            address: address as u64,
            length: length as u64,
            flags: if writable { 0xc } else { 0xd },
        })
    });
    let _restore = Restore(previous);
    run()
}

#[repr(C)]
struct Userptr {
    address: u64,
    length: u64,
    flags: u32,
    handle: u32,
}
type DrmCall =
    unsafe extern "C" fn(libc::c_int, libc::c_ulong, *mut c_void, libc::c_ulong) -> libc::c_int;

// The library remains loaded for the process lifetime. Initialization is shared
// safely, unlike an unsynchronized cache of the fallback dlopen handle.
fn real_call() -> Option<DrmCall> {
    static REAL: OnceLock<usize> = OnceLock::new();
    let address = *REAL.get_or_init(|| unsafe {
        let mut symbol = libc::dlsym(libc::RTLD_NEXT, c"drmCommandWriteRead".as_ptr());
        if symbol.is_null() {
            let library = libc::dlopen(c"libdrm.so.2".as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
            if !library.is_null() {
                symbol = libc::dlsym(library, c"drmCommandWriteRead".as_ptr());
            }
        }
        symbol as usize
    });
    if address == 0 {
        None
    } else {
        Some(unsafe { std::mem::transmute::<usize, DrmCall>(address) })
    }
}

/// libdrm ABI entry point exported by the plugin executable.
///
/// # Safety
/// `data` must satisfy libdrm's contract for `command` and `length`.
#[unsafe(export_name = "drmCommandWriteRead")]
pub unsafe extern "C" fn drm_command_write_read(
    fd: libc::c_int,
    command: libc::c_ulong,
    data: *mut c_void,
    length: libc::c_ulong,
) -> libc::c_int {
    if command == 0x11 && length as usize == std::mem::size_of::<Userptr>() && !data.is_null() {
        let _ = INTENT.try_with(|cell| {
            let intent = cell.get();
            let args = unsafe { &mut *data.cast::<Userptr>() };
            if intent.address != 0 && args.address == intent.address && args.length <= intent.length
            {
                args.flags = intent.flags;
            }
        });
    }
    match real_call() {
        Some(call) => unsafe { call(fd, command, data, length) },
        None => -libc::ENOSYS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn flags() -> u32 {
        let mut args = Userptr {
            address: 0x1000,
            length: 4096,
            flags: 0xe,
            handle: 0,
        };
        unsafe {
            drm_command_write_read(-1, 0x11, (&mut args as *mut Userptr).cast(), 24);
        }
        args.flags
    }
    #[test]
    fn only_opted_in_thread_and_scope_changes() {
        assert_eq!(flags(), 0xe);
        with_import(0x1000, 4096, false, || {
            assert_eq!(flags(), 0xd);
            assert_eq!(std::thread::spawn(flags).join().unwrap(), 0xe);
            with_import(0x1000, 4096, true, || assert_eq!(flags(), 0xc));
            assert_eq!(flags(), 0xd);
        });
        assert_eq!(flags(), 0xe);
        with_import(0x2000, 4096, true, || assert_eq!(flags(), 0xe));
    }
    #[test]
    fn restores_on_unwind() {
        let _ = std::panic::catch_unwind(|| with_import(0x1000, 4096, true, || panic!("test")));
        assert_eq!(flags(), 0xe);
    }
}
