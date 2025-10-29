use no_std_strings::{str_format, str256};

use crate::{
    constants::*,
    host_calls::{host_log, host_terminate, host_write},
    kernel::print,
    linux_abi::Err,
    p9::{
        P9Response, P9SetattrMask, Qid, RattachMessage, RclunkMessage, ReadableMessage,
        RgetattrMessage, RlcreateMessage, RlopenMessage, RmkdirMessage, RmknodMessage,
        RreadMessage, RreaddirMessage, RreadlinkMessage, RremoveMessage, RsetattrMessage,
        RsymlinkMessage, RversionMessage, RwalkMessage, RwriteMessage, TattachMessage,
        TlcreateMessage, TlopenMessage, TmkdirMessage, TmknodMessage, TreadMessage,
        TreaddirMessage, TreadlinkMessage, TremoveMessage, TsetattrMessage, TsymlinkMessage,
        TversionMessage, TwriteMessage, constants::*,
    },
};

#[cfg(target_arch = "riscv32")]
use alloc::format;
#[cfg(target_arch = "riscv32")]
use alloc::string::String;
#[cfg(target_arch = "riscv32")]
use alloc::string::ToString;
#[cfg(target_arch = "riscv32")]
use alloc::vec;
#[cfg(target_arch = "riscv32")]
use alloc::vec::Vec;
#[cfg(target_arch = "riscv32")]
use core::str;

// Filesystem-related syscalls

// fcntl constants
#[allow(dead_code)]
pub const F_DUPFD: u32 = 0; // Duplicate file descriptor.
#[allow(dead_code)]
pub const F_GETFD: u32 = 1; // Get file descriptor flags.
pub const F_SETFD: u32 = 2; // Set file descriptor flags.
#[allow(dead_code)]
pub const F_GETFL: u32 = 3; // Get file status flags.
#[allow(dead_code)]
pub const F_SETFL: u32 = 4; // Set file status flags.
pub const FD_CLOEXEC: u32 = 1; // Close on exec flag.

// File locking constants
#[allow(dead_code)]
pub const F_GETLK: u32 = 12; // Get file lock
#[allow(dead_code)]
pub const F_SETLK: u32 = 13; // Set file lock
#[allow(dead_code)]
pub const F_SETLKW: u32 = 14; // Set file lock (wait)

// Lock types
#[allow(dead_code)]
pub const F_RDLCK: u32 = 0; // Read lock
#[allow(dead_code)]
pub const F_WRLCK: u32 = 1; // Write lock
#[allow(dead_code)]
pub const F_UNLCK: u32 = 2; // Unlock

pub static mut ROOT_FID: u32 = 0;
pub static mut CWD_FID: u32 = 0;
pub static mut CWD_STR: String = String::new();

pub fn get_root_fid() -> u32 {
    unsafe { ROOT_FID }
}

#[allow(dead_code)]
pub fn get_cwd_fid() -> u32 {
    unsafe { CWD_FID }
}

#[allow(dead_code)]
pub fn set_root_fid(fid: u32) {
    unsafe {
        ROOT_FID = fid;
    }
}

#[allow(dead_code)]
pub fn set_cwd_fid(fid: u32) {
    unsafe {
        CWD_FID = fid;
    }
}

#[allow(dead_code)]
pub fn set_cwd_str(cwd_str: String) {
    unsafe {
        CWD_STR = cwd_str;
    }
}

#[allow(dead_code)]
pub fn get_cwd_str() -> String {
    #[allow(static_mut_refs)]
    unsafe {
        CWD_STR.clone()
    }
}

// File description (like Linux's struct file) - shared state for duplicated fds
#[derive(Copy, Clone)]
pub struct FileDescription {
    pub fid: u32,
    pub cursor: u64, // Shared cursor for all fds pointing to this description
    pub is_dir: bool,
    pub mode: u32,
    pub flags: u32,    // Linux open flags (O_APPEND, etc.)
    pub refcount: u32, // Number of file descriptors pointing to this description
}

// File descriptor table - each fd points to a file description
#[derive(Copy, Clone)]
pub struct FileDescriptor {
    pub file_desc_id: u32, // Index into FILE_DESC_TABLE, 0xFF means unused
    pub cloexec: bool,     // FD_CLOEXEC flag (close-on-exec)
}

// File description table (similar to Linux's file table)
pub static mut FILE_DESC_TABLE: [FileDescription; 256] = [FileDescription {
    fid: 0,
    cursor: 0,
    is_dir: false,
    mode: 0xFFFF_FFFF,
    flags: 0,
    refcount: 0,
}; 256];

// File descriptor table (maps fd -> file_desc_id)
pub static mut FD_TABLE: [FileDescriptor; 256] = [FileDescriptor {
    file_desc_id: 0xFF, // 0xFF means unused
    cloexec: false,
}; 256];

pub fn get_fd(fd: u32) -> FileDescriptor {
    unsafe { FD_TABLE[fd as usize] }
}

pub fn update_fd(fd: u32, file_descriptor: FileDescriptor) {
    unsafe {
        FD_TABLE[fd as usize] = file_descriptor;
    }
}

// Get file description by ID
pub fn get_file_desc(desc_id: u32) -> FileDescription {
    unsafe { FILE_DESC_TABLE[desc_id as usize] }
}

// Update file description
pub fn update_file_desc(desc_id: u32, desc: FileDescription) {
    unsafe {
        FILE_DESC_TABLE[desc_id as usize] = desc;
    }
}

// Helper to get file description from file descriptor
#[allow(dead_code)]
pub fn get_file_desc_from_fd(fd: u32) -> Option<FileDescription> {
    let fd_entry = get_fd(fd);
    if fd_entry.file_desc_id == 0xFF {
        None
    } else {
        Some(get_file_desc(fd_entry.file_desc_id))
    }
}

// Helper to update file description through file descriptor
#[allow(dead_code)]
pub fn update_file_desc_through_fd(fd: u32, desc: FileDescription) -> Result<(), Err> {
    let fd_entry = get_fd(fd);
    if fd_entry.file_desc_id == 0xFF {
        return Err(Err::BadFd);
    }
    update_file_desc(fd_entry.file_desc_id, desc);
    Ok(())
}

// Find a free file description slot
pub fn find_free_file_desc() -> Result<u32, Err> {
    for i in 0..256 {
        let desc = get_file_desc(i);
        if desc.fid == 0 && desc.refcount == 0 {
            return Ok(i);
        }
    }
    kprint!("find_free_file_desc: no free file descriptions available");
    Err(Err::MFile)
}

// Increment reference count for a file description
pub fn inc_refcount(desc_id: u32) {
    let mut desc = get_file_desc(desc_id);
    desc.refcount += 1;
    update_file_desc(desc_id, desc);
}

// Decrement reference count for a file description
// Returns true if this was the last reference (should clunk)
pub fn dec_refcount(desc_id: u32) -> bool {
    let mut desc = get_file_desc(desc_id);
    if desc.refcount > 0 {
        desc.refcount -= 1;
    }
    let should_clunk = desc.refcount == 0;
    update_file_desc(desc_id, desc);
    should_clunk
}

// Linux dirent64 structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct LinuxDirent64 {
    d_ino: u64,      // inode number
    d_off: i64,      // offset to next record
    d_reclen: u16,   // length of this record
    d_type: u8,      // file type
    d_name: [u8; 1], // filename (variable length, but we use [u8; 1] for the struct)
}

// Directory entry types
#[allow(dead_code)]
pub const DT_UNKNOWN: u8 = 0;
#[allow(dead_code)]
const DT_FIFO: u8 = 1;
#[allow(dead_code)]
const DT_CHR: u8 = 2;
pub const DT_DIR: u8 = 4;
#[allow(dead_code)]
const DT_BLK: u8 = 6;
pub const DT_REG: u8 = 8;
pub const DT_LNK: u8 = 10;
#[allow(dead_code)]
const DT_SOCK: u8 = 12;

// make this a global static that we can set
static mut P9_ENABLED: bool = false;

pub fn set_p9_enabled(enabled: bool) {
    unsafe {
        P9_ENABLED = enabled;
    }
}

pub fn get_p9_enabled() -> bool {
    unsafe { P9_ENABLED }
}

pub fn sys_unlinkat(dfd: u32, pathname: u32, flag: u32) -> Result<u32, Err> {
    if !get_p9_enabled() {
        let msg = b"sys_unlinkat: p9 is not enabled";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }

    // Read the pathname from user space
    let filename = unsafe { core::slice::from_raw_parts(pathname as *const u8, 256) };
    let null_pos = filename
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(filename.len());
    let filename = &filename[..null_pos];
    let filename_str = match str::from_utf8(filename) {
        Ok(s) => s,
        Err(_) => {
            kprint!("sys_unlinkat: invalid UTF-8 filename");
            return Err(Err::NoSys);
        }
    };

    kprint!(
        "sys_unlinkat: dfd={}, filename='{}', flag={}",
        dfd,
        filename_str,
        flag
    );

    resolve_path(dfd, filename_str, 0xFFFF_FFFE)?;
    let tremove = TremoveMessage::new(0, 0xFFFF_FFFE);
    match tremove.send_tremove() {
        Ok(bytes_written) => {
            kprint!("sys_unlinkat: sent {} bytes for Tremove", bytes_written);
        }
        Err(e) => {
            kprint!("sys_unlinkat: error sending Tremove: {:?}", e);
            return Err(Err::NoSys);
        }
    }

    match RremoveMessage::read_response() {
        P9Response::Success(rremove) => {
            kprint!("sys_unlinkat: rremove = {:?}", rremove);
            clunk(0xFFFF_FFFE, false);
            Ok(0)
        }
        P9Response::Error(rlerror2) => {
            kprint!(
                "sys_unlinkat: received Rlerror for Rremove: tag={}, ecode={}",
                rlerror2.tag,
                rlerror2.ecode
            );
            Ok(-(rlerror2.ecode as i32) as u32)
        }
    }
}

// Statfs64 structure matching the Linux kernel definition
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Statfs64 {
    f_type: u32,       // Filesystem type
    f_bsize: u32,      // Block size
    f_blocks: u64,     // Total blocks
    f_bfree: u64,      // Free blocks
    f_bavail: u64,     // Available blocks
    f_files: u64,      // Total inodes
    f_ffree: u64,      // Free inodes
    f_fsid: u64,       // Filesystem ID (as u64 for simplicity)
    f_namelen: u32,    // Maximum filename length
    f_frsize: u32,     // Fragment size
    f_flags: u32,      // Mount flags
    f_spare: [u32; 4], // Spare fields
}

pub fn sys_statfs64(path: u32, sz: u32, buf: u32) -> Result<u32, Err> {
    if !get_p9_enabled() {
        let msg = b"sys_statfs64: p9 is not enabled";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }

    kprint!("sys_statfs64: path={}, sz={}, buf={}", path, sz, buf);

    // Validate the size parameter
    if sz != core::mem::size_of::<Statfs64>() as u32 {
        kprint!(
            "sys_statfs64: invalid size parameter: expected {}, got {}",
            core::mem::size_of::<Statfs64>(),
            sz
        );
        return Err(Err::NoSys);
    }

    // Create statfs64 structure with the specified f_type
    let statfs = Statfs64 {
        f_type: 0x01021994,         // 9P filesystem type as requested
        f_bsize: 4096,              // 4KB block size (typical for 9P)
        f_blocks: 1000000,          // Total blocks (placeholder)
        f_bfree: 500000,            // Free blocks (placeholder)
        f_bavail: 500000,           // Available blocks (placeholder)
        f_files: 10000,             // Total inodes (placeholder)
        f_ffree: 5000,              // Free inodes (placeholder)
        f_fsid: 0x1234567890ABCDEF, // Filesystem ID (placeholder)
        f_namelen: 255,             // Maximum filename length
        f_frsize: 4096,             // Fragment size (same as block size)
        f_flags: 0,                 // No special mount flags
        f_spare: [0; 4],            // Spare fields
    };

    // Write the statfs64 structure to user memory using copy_to_user
    let statfs_bytes = unsafe {
        core::slice::from_raw_parts(
            &statfs as *const Statfs64 as *const u8,
            core::mem::size_of::<Statfs64>(),
        )
    };
    let bytes_copied = crate::kernel::copy_to_user(
        buf as *mut u8,
        statfs_bytes.as_ptr(),
        core::mem::size_of::<Statfs64>(),
    );
    if bytes_copied == 0 {
        kprint!("sys_statfs64: failed to copy statfs64 structure to user memory");
        return Err(Err::NoSys);
    }

    kprint!(
        "sys_statfs64: wrote statfs64 structure with f_type=0x{:x}",
        statfs.f_type
    );
    Ok(0)
}

pub fn init_fs() {
    set_fd(0, 0xFFFF_FFFE);
    set_fd(1, 0xFFFF_FFFE);
    set_fd(2, 0xFFFF_FFFE);
}

pub fn sys_close(_fd: u32) -> Result<u32, Err> {
    if _fd >= 256 {
        return Err(Err::BadFd);
    }
    let fd_entry = get_fd(_fd);
    if fd_entry.file_desc_id == 0xFF {
        // File descriptor is not open
        return Err(Err::BadFd);
    }

    let desc_id = fd_entry.file_desc_id;
    let file_desc = get_file_desc(desc_id);

    // Decrement reference count
    let should_clunk = dec_refcount(desc_id);

    // Only clunk the fid when this is the last reference
    if should_clunk && file_desc.fid != 0xFFFF_FFFE && file_desc.fid != 0 {
        clunk(file_desc.fid, false);
        // Clear the file description
        update_file_desc(
            desc_id,
            FileDescription {
                fid: 0,
                cursor: 0,
                is_dir: false,
                mode: 0xFFFF_FFFF,
                flags: 0,
                refcount: 0,
            },
        );
    }

    // Clear the file descriptor entry
    update_fd(
        _fd,
        FileDescriptor {
            file_desc_id: 0xFF,
            cloexec: false,
        },
    );

    Ok(0)
}

pub fn read_file_to_user_memory(fd: u32, buf: u32, count: u32, offset: u64) -> Result<u32, Err> {
    if fd >= 256 {
        return Err(Err::Inval);
    }
    let fd_entry = get_fd(fd);
    if fd_entry.file_desc_id == 0xFF {
        return Err(Err::BadFd);
    }

    let file_desc = get_file_desc(fd_entry.file_desc_id);
    if file_desc.fid != 0 {
        let mut cursor = offset;
        let mut total_bytes_read = 0u32;
        while cursor < offset + (count as u64) {
            // read max 128 bytes at a time
            let max_count = (offset + count as u64 - cursor).min(128) as u32;
            let tread = TreadMessage::new(0, file_desc.fid, cursor, max_count);
            match tread.send_tread() {
                Ok(_bytes_written) => {
                    // Success
                }
                Err(_e) => {
                    kprint!("read_file_to_user_memory: error sending tread: {:?}", _e);
                    return Err(Err::NoSys);
                }
            }
            match RreadMessage::read_response() {
                P9Response::Success(rread) => {
                    if rread.count == 0 {
                        kprint!(
                            "read_file_to_user_memory: read 0 bytes, offset={} cursor={} count={}, max_count={}",
                            offset,
                            cursor,
                            count,
                            max_count
                        );
                        return Ok(total_bytes_read);
                    }
                    let user_ptr = buf as *mut u8;
                    let user_ptr = unsafe { user_ptr.add(cursor as usize).sub(offset as usize) };
                    let data = rread.data;
                    let _bytes_copied =
                        crate::kernel::copy_to_user(user_ptr, data.as_ptr(), rread.count as usize);
                    cursor += rread.count as u64;
                    total_bytes_read += rread.count;
                }
                P9Response::Error(_rlerror) => {
                    if _rlerror.ecode == 0 {
                        return Ok(total_bytes_read);
                    }
                    return Err(Err::NoSys);
                }
            }
        }
        return Ok(total_bytes_read);
    }
    Err(Err::NoSys)
}

pub fn sys_read(_fd: u32, _buf: u32, _count: u32) -> Result<u32, Err> {
    // Check for NULL buffer
    if _buf == 0 {
        return Err(Err::Fault);
    }

    if !get_p9_enabled() {
        let msg = b"sys_read: p9 is not enabled";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }

    // Validate file descriptor is within range
    if _fd >= 256 {
        kprint!("sys_read: invalid fd={}", _fd);
        return Err(Err::BadFd);
    }

    kprint!("sys_read: _fd={} _buf={} _count={}", _fd, _buf, _count);
    let fd_entry = get_fd(_fd);

    // Check if fd is closed/invalid
    if fd_entry.file_desc_id == 0xFF {
        kprint!("sys_read: fd={} is not open", _fd);
        return Err(Err::BadFd);
    }

    let file_desc = get_file_desc(fd_entry.file_desc_id);

    // Check if fd is a directory
    if file_desc.is_dir {
        kprint!("sys_read: fd={} is a directory", _fd);
        return Err(Err::IsDir);
    }

    if file_desc.fid != 0 && file_desc.fid != 0xFFFF_FFFE {
        // read with Tread and Rread from the fid using 9p protocol and update .cursor in FILE_DESC_TABLE
        let tread = TreadMessage::new(0, file_desc.fid, file_desc.cursor, _count);
        match tread.send_tread() {
            Ok(_bytes_written) => {
                // Success
            }
            Err(_e) => {
                return Err(Err::NoSys);
            }
        }
        match RreadMessage::read_response() {
            P9Response::Success(rread) => {
                kprint!("sys_read: received {} bytes from 9P", rread.count);
                let user_ptr = _buf as *mut u8;
                let data = rread.data;

                if rread.count == 0 {
                    kprint!("sys_read: read 0 bytes (EOF)");
                    return Ok(0);
                }

                kprint!(
                    "sys_read: copying {} bytes to user buffer 0x{:08x}",
                    rread.count,
                    _buf
                );
                let bytes_copied =
                    crate::kernel::copy_to_user(user_ptr, data.as_ptr(), rread.count as usize);

                if bytes_copied == 0 {
                    kprint!(
                        "sys_read: copy_to_user failed! dst=0x{:08x}, len={}",
                        _buf,
                        rread.count
                    );
                    return Err(Err::NoSys);
                }
                // Update the shared cursor in the file description
                let mut updated_desc = file_desc;
                updated_desc.cursor += rread.count as u64;
                update_file_desc(fd_entry.file_desc_id, updated_desc);
                return Ok(rread.count);
            }
            P9Response::Error(_rlerror) => {
                return Err(Err::NoSys);
            }
        }
    }
    let msg = b"sys_read not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_write(fd: u32, buf: u32, count: u32) -> Result<u32, Err> {
    do_write(fd as i32, buf as *const u8, count as usize).map(|x| x as u32)
}

pub fn do_write(fd: i32, buf: *const u8, count: usize) -> Result<usize, Err> {
    // Special case: NULL buffer with count=0 is allowed (Linux behavior)
    // This allows write(fd, NULL, 0) to succeed and return 0
    // Note: On Linux, write(fd, NULL, -1) (which becomes count=0xFFFFFFFF after cast)
    // will also be treated as 0 if count would overflow, but we handle count=0 explicitly
    if buf.is_null() && count == 0 {
        return Ok(0);
    }

    // NULL buffer with count > 0 is invalid (causes EFAULT)
    if buf.is_null() {
        return Err(Err::Fault);
    }

    if fd == 1 || fd == 2 {
        host_write(fd as u32, buf, count);
    } else {
        if !(0..256).contains(&fd) {
            return Err(Err::Inval);
        }
        if !get_p9_enabled() {
            let msg = b"do_write: p9 is not enabled";
            host_log(msg.as_ptr(), msg.len());
            return Err(Err::NoSys);
        }
        let fd_entry = get_fd(fd as u32);
        if fd_entry.file_desc_id == 0xFF {
            let msg = b"do_write: fd is not open";
            host_log(msg.as_ptr(), msg.len());
            return Err(Err::BadFd);
        }

        let mut file_desc = get_file_desc(fd_entry.file_desc_id);
        if file_desc.fid != 0 && file_desc.fid != 0xFFFF_FFFE {
            // Handle O_APPEND: position at end of file before each write
            const O_APPEND: u32 = 0o2000;
            let mut current_cursor = file_desc.cursor;

            if (file_desc.flags & O_APPEND) == O_APPEND {
                // Get file size using getattr
                let tgetattr = crate::p9::TgetattrMessage::new(0, file_desc.fid, P9_GETATTR_SIZE);
                match tgetattr.send_tgetattr() {
                    Ok(_) => match RgetattrMessage::read_response() {
                        P9Response::Success(rgetattr) => {
                            current_cursor = rgetattr.size;
                            kprint!(
                                "do_write: O_APPEND - setting cursor to end of file: {}",
                                current_cursor
                            );
                        }
                        P9Response::Error(_) => {
                            kprint!("do_write: error getting file size for O_APPEND");
                            return Err(Err::NoSys);
                        }
                    },
                    Err(_) => {
                        kprint!("do_write: error sending getattr for O_APPEND");
                        return Err(Err::NoSys);
                    }
                }
            }

            // Write in chunks of 512 bytes
            const CHUNK_SIZE: usize = 512;
            let mut total_written = 0;

            let data = unsafe { core::slice::from_raw_parts(buf, count) };

            for chunk_start in (0..count).step_by(CHUNK_SIZE) {
                let chunk_end = (chunk_start + CHUNK_SIZE).min(count);
                let chunk_data = &data[chunk_start..chunk_end];

                kprint!(
                    "do_write: writing chunk {} bytes at cursor {}",
                    chunk_data.len(),
                    current_cursor
                );

                let twrite =
                    TwriteMessage::new(0, file_desc.fid, current_cursor, chunk_data.to_vec());

                match twrite.send_twrite() {
                    Ok(_bytes_written) => {
                        // Success
                    }
                    Err(_e) => {
                        kprint!("do_write: error sending Twrite: {:?}", _e);
                        return Err(Err::NoSys);
                    }
                }

                match RwriteMessage::read_response() {
                    P9Response::Success(rwrite) => {
                        let bytes_written = rwrite.count as usize;
                        total_written += bytes_written;
                        current_cursor += bytes_written as u64;

                        kprint!(
                            "do_write: wrote {} bytes, total: {}",
                            bytes_written,
                            total_written
                        );

                        // If we wrote fewer bytes than requested, we're done
                        if bytes_written < chunk_data.len() {
                            break;
                        }
                    }
                    P9Response::Error(_rlerror) => {
                        kprint!("do_write: error sending Rwrite: {:?}", _rlerror);
                        return Err(Err::NoSys);
                    }
                }
            }

            // Update the shared cursor in the file description
            file_desc.cursor = current_cursor;
            update_file_desc(fd_entry.file_desc_id, file_desc);

            return Ok(total_written);
        }
        let msg = b"do_write for fd > 2 that are not FIDs not implemented";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }

    Ok(count)
}

pub fn sys_writev(fd: u32, vec_ptr: u32, vlen: u32) -> Result<u32, Err> {
    let fd = fd as i32;
    let vec_ptr = vec_ptr as *const IoVec;
    let vec = unsafe { core::slice::from_raw_parts(vec_ptr, vlen as usize) };

    let mut total: usize = 0;
    for iov in vec {
        total += do_write(fd, iov.iov_base, iov.iov_len)?;
    }
    Ok(total as u32)
}

pub fn sys_chdir(filename: u32) -> Result<u32, Err> {
    if !get_p9_enabled() {
        let msg = b"sys_chdir: p9 is not enabled";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }

    let filename = unsafe { core::slice::from_raw_parts(filename as *const u8, 256) };
    kprint!("sys_chdir: filename='{:?}'", filename);

    let null_pos = filename
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(filename.len());
    let filename = &filename[..null_pos];
    let filename_str = match str::from_utf8(filename) {
        Ok(s) => s,
        Err(_) => {
            kprint!("sys_chdir: invalid UTF-8 filename");
            return Err(Err::NoSys);
        }
    };

    kprint!("sys_chdir: filename='{}'", filename_str);

    let starting_fid = if filename_str.starts_with("/") {
        get_root_fid()
    } else {
        get_cwd_fid()
    };
    let wnames = normalize_path(filename_str);
    let walk_result = do_walk(starting_fid, 0xFFFF_FFFC, wnames);
    match walk_result {
        Ok((ret_code, _)) => {
            if ret_code != 0 {
                return Ok(ret_code);
            }
        }
        Err(e) => {
            kprint!("sys_chdir: error walking to directory: {:?}", e.as_errno());
            return Err(e);
        }
    }
    if get_cwd_fid() == 0xFFFF_FFFD {
        clunk(0xFFFF_FFFD, true); // remove the old cwd fix - allow CWD clunking
    }

    // Build the new path correctly
    let new_cwd_str = if filename_str.starts_with("/") {
        // Absolute path - use as-is
        filename_str.to_string()
    } else {
        // Relative path - append to current directory
        let current_cwd = get_cwd_str();
        if current_cwd.ends_with("/") {
            format!("{}{}", current_cwd, filename_str)
        } else {
            format!("{}/{}", current_cwd, filename_str)
        }
    };

    // Resolve ..'s and .'s properly
    let path_components = normalize_path(&new_cwd_str);
    let mut resolved_path = Vec::new();

    for component in path_components {
        if component == ".." {
            if !resolved_path.is_empty() {
                resolved_path.pop();
            }
            // If we're at root and encounter .., stay at root
        } else if component != "." && !component.is_empty() {
            resolved_path.push(component);
        }
    }

    // Build final path
    let mut final_path = if resolved_path.is_empty() {
        "/".to_string()
    } else {
        format!("/{}", resolved_path.join("/"))
    };

    // Ensure it ends with / for directories
    if !final_path.ends_with("/") {
        final_path.push('/');
    }

    kprint!("sys_chdir: new_cwd_str='{}'", final_path);
    set_cwd_str(final_path);
    dup_fid_to(0xFFFF_FFFC, 0xFFFF_FFFD)?;
    clunk(0xFFFF_FFFC, true); // allow CWD clunking in sys_chdir
    set_cwd_fid(0xFFFF_FFFD);
    Ok(0)
}

fn dup_fid_to(fid: u32, fid_to: u32) -> Result<u32, Err> {
    let walk_result = do_walk(fid, fid_to, vec![]);
    match walk_result {
        Ok(_) => {
            // Walk succeeded, return the new fid (fid_to), not the return code
            Ok(fid_to)
        }
        Err(e) => {
            kprint!("dup_fid_to: error dupping: {:?}", e.as_errno());
            clunk(fid_to, false);
            Err(e)
        }
    }
}

pub fn sys_dup(fd: u32) -> Result<u32, Err> {
    if !get_p9_enabled() {
        let msg = b"sys_dup: p9 is not enabled";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }
    // Check if fd is out of range (including negative values cast to u32)
    if fd >= 256 {
        kprint!("sys_dup: fd {} is out of range", fd);
        return Err(Err::BadFd);
    }
    // diod/p9 protocol "fid can be cloned to newfid by calling walk with nwname set to zero."
    let fd_entry = get_fd(fd);
    if fd_entry.file_desc_id == 0xFF {
        kprint!("sys_dup: fd {} is not open", fd);
        return Err(Err::BadFd);
    }

    let file_desc = get_file_desc(fd_entry.file_desc_id);

    // Allocate a new file descriptor (not a new file description!)
    let new_fd = find_free_fd()?;

    // Make the new fd point to the same file description
    // sys_dup copies the FD_CLOEXEC flag from the original fd
    update_fd(
        new_fd,
        FileDescriptor {
            file_desc_id: fd_entry.file_desc_id,
            cloexec: fd_entry.cloexec,
        },
    );

    // Increment the reference count for the shared file description
    inc_refcount(fd_entry.file_desc_id);

    kprint!(
        "sys_dup: duplicated fd {} to fd {}, both pointing to desc_id {} (fid={}, refcount now {})",
        fd,
        new_fd,
        fd_entry.file_desc_id,
        file_desc.fid,
        get_file_desc(fd_entry.file_desc_id).refcount
    );

    Ok(new_fd)
}

pub fn sys_dup3(oldfd: u32, newfd: u32, flags: u32) -> Result<u32, Err> {
    if !get_p9_enabled() {
        let msg = b"sys_dup3: p9 is not enabled";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }

    kprint!(
        "sys_dup3: oldfd={}, newfd={}, flags={}",
        oldfd,
        newfd,
        flags
    );

    // Validate flags FIRST (O_CLOEXEC is the only valid flag for dup3)
    // Invalid flags should return EINVAL even if fds are invalid
    const O_CLOEXEC: u32 = 0o2000000;
    if (flags & !O_CLOEXEC) != 0 {
        kprint!("sys_dup3: invalid flags 0x{:x}", flags);
        return Err(Err::Inval);
    }

    // Validate file descriptors are in range
    if oldfd >= 256 {
        kprint!("sys_dup3: oldfd {} is out of range", oldfd);
        return Err(Err::BadFd);
    }
    if newfd >= 256 {
        kprint!("sys_dup3: newfd {} is out of range", newfd);
        return Err(Err::BadFd);
    }

    // Check if oldfd is the same as newfd (not allowed in dup3, unlike dup2)
    if oldfd == newfd {
        kprint!("sys_dup3: oldfd == newfd (not allowed in dup3)");
        return Err(Err::Inval);
    }

    // Check if oldfd is open
    let old_fd_entry = get_fd(oldfd);
    if old_fd_entry.file_desc_id == 0xFF {
        kprint!("sys_dup3: oldfd {} is not open", oldfd);
        return Err(Err::BadFd);
    }

    // Close newfd if it's currently open
    let new_fd_entry = get_fd(newfd);
    if new_fd_entry.file_desc_id != 0xFF {
        kprint!(
            "sys_dup3: closing existing fd {} (desc_id={})",
            newfd,
            new_fd_entry.file_desc_id
        );
        let _ = sys_close(newfd); // Ignore errors from close
    }

    // Make newfd point to the same file description as oldfd
    // Set FD_CLOEXEC based on O_CLOEXEC flag
    let cloexec_set = (flags & O_CLOEXEC) != 0;
    update_fd(
        newfd,
        FileDescriptor {
            file_desc_id: old_fd_entry.file_desc_id,
            cloexec: cloexec_set,
        },
    );

    // Increment the reference count for the shared file description
    inc_refcount(old_fd_entry.file_desc_id);

    let file_desc = get_file_desc(old_fd_entry.file_desc_id);
    kprint!(
        "sys_dup3: duplicated fd {} to fd {}, both pointing to desc_id {} (fid={}, refcount now {})",
        oldfd,
        newfd,
        old_fd_entry.file_desc_id,
        file_desc.fid,
        get_file_desc(old_fd_entry.file_desc_id).refcount
    );

    Ok(newfd)
}

pub fn sys_faccessat(_dfd: u32, _filename: u32, _mode: u32) -> Result<u32, Err> {
    // faccessat calls faccessat2 with no flags
    sys_faccessat2(_dfd, _filename, _mode, 0)
}

pub fn sys_faccessat2(_dfd: u32, _filename: u32, _mode: u32, _flags: u32) -> Result<u32, Err> {
    // faccessat2 checks if a file can be accessed with the given mode
    // mode bits: R_OK (4), W_OK (2), X_OK (1), F_OK (0 - existence only)
    // Flags: AT_EACCESS (0x200), AT_SYMLINK_NOFOLLOW (0x100), AT_EMPTY_PATH (0x1000)

    if !get_p9_enabled() {
        let msg = b"sys_faccessat2: p9 is not enabled";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }

    // Check for invalid address (NULL pointer or out of user memory range)
    if _filename == 0 {
        kprint!("sys_faccessat2: invalid filename address (NULL)");
        return Err(Err::Fault);
    }

    // Check if address is in user memory range (below 0xC0000000)
    if _filename >= 0xC0000000 {
        kprint!(
            "sys_faccessat2: invalid filename address (out of user memory): 0x{:x}",
            _filename
        );
        return Err(Err::Fault);
    }

    // Validate flags - check for invalid flag values
    const AT_EACCESS: u32 = 0x200;
    const AT_SYMLINK_NOFOLLOW: u32 = 0x100;
    const AT_EMPTY_PATH: u32 = 0x1000;
    const AT_FLAGS_MASK: u32 = AT_EACCESS | AT_SYMLINK_NOFOLLOW | AT_EMPTY_PATH;

    // Check for invalid flags
    // -1 (0xFFFFFFFF when cast from i32) sets all bits, which is invalid
    let flags_i32 = _flags as i32;
    if flags_i32 < 0 || (_flags & !AT_FLAGS_MASK) != 0 {
        kprint!("sys_faccessat2: invalid flags 0x{:x}", _flags);
        return Err(Err::Inval);
    }

    // Validate mode - check for invalid mode values
    // Valid modes: F_OK (0), R_OK (4), W_OK (2), X_OK (1), or combinations
    // -1 (0xFFFFFFFF when cast from i32) is invalid
    let mode_i32 = _mode as i32;
    const MODE_MASK: u32 = 0x7; // Only lowest 3 bits valid
    if mode_i32 < 0 || (_mode & !MODE_MASK) != 0 {
        kprint!("sys_faccessat2: invalid mode 0x{:x}", _mode);
        return Err(Err::Inval);
    }

    // Parse filename
    let filename = unsafe { core::slice::from_raw_parts(_filename as *const u8, 256) };
    let null_pos = filename
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(filename.len());
    let filename_str = str::from_utf8(&filename[..null_pos]).map_err(|_| Err::Inval)?;

    kprint!(
        "sys_faccessat2: dfd={}, filename='{}', mode=0x{:x}, flags=0x{:x}",
        _dfd,
        filename_str,
        _mode,
        _flags
    );

    if (_flags & AT_EACCESS) != 0 {
        kprint!("sys_faccessat2: AT_EACCESS set (using effective uid/gid)");
        // In zkVM context, we don't distinguish between real and effective uid/gid
    }

    if (_flags & AT_SYMLINK_NOFOLLOW) != 0 {
        kprint!("sys_faccessat2: AT_SYMLINK_NOFOLLOW set");
        // TODO: when checking access, don't follow symlinks
    }

    // Handle AT_EMPTY_PATH - check access on the fd itself
    if (_flags & AT_EMPTY_PATH) != 0 && filename_str.is_empty() {
        kprint!("sys_faccessat2: AT_EMPTY_PATH set, checking fd {}", _dfd);

        // Validate dfd is in range
        if _dfd >= 256 {
            return Err(Err::BadFd);
        }

        let fd_entry = get_fd(_dfd);
        if fd_entry.file_desc_id == 0xFF {
            // Invalid file descriptor
            return Err(Err::BadFd);
        }

        // File descriptor is valid and open - access granted
        Ok(0)
    } else {
        // Normal path-based access check
        // Use temp fids that don't conflict with CWD (0xFFFF_FFFD)
        let starting_fid = get_starting_fid(_dfd, filename_str)?;
        let (dir_path, file_name) = split_path(filename_str);
        let dir_path = normalize_path(&dir_path);
        do_walk(starting_fid, 0xFFFF_FFFB, dir_path)?;

        match resolve_file_to_fid(0xFFFF_FFFB, 0xFFFF_FFFA, &file_name) {
            Ok(_) => {
                // File exists and is accessible
                clunk(0xFFFF_FFFA, false);
                clunk(0xFFFF_FFFB, false);
                Ok(0)
            }
            Err(e) => {
                // File not found or not accessible
                clunk(0xFFFF_FFFB, false);
                Err(e)
            }
        }
    }
}

pub fn sys_fadvise64_64(_fd: u32, _offset: u32, _len: u32, _advice: u32) -> Result<u32, Err> {
    let msg = b"sys_fadvise64_64 not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_fallocate(
    fd: u32,
    mode: u32,
    offset_low: u32,
    offset_high: u32,
    len_low: u32,
    len_high: u32,
) -> Result<u32, Err> {
    // On 32-bit RISC-V, 64-bit arguments are passed in two registers (low, high)
    // Check for negative values BEFORE converting to u64
    // If the high word has the sign bit set (0x80000000), it's negative
    let offset_i64 = ((offset_high as i32 as i64) << 32) | (offset_low as i64);
    let len_i64 = ((len_high as i32 as i64) << 32) | (len_low as i64);

    // Validate parameters before proceeding
    if offset_i64 < 0 {
        kprint!(
            "sys_fallocate: negative offset {}, returning EINVAL",
            offset_i64
        );
        return Err(Err::Inval);
    }

    if len_i64 <= 0 {
        kprint!(
            "sys_fallocate: invalid len {} (must be > 0), returning EINVAL",
            len_i64
        );
        return Err(Err::Inval);
    }

    // Now convert to u64 for calculations
    let offset: u64 = offset_i64 as u64;
    let len: u64 = len_i64 as u64;
    let end_pos = offset + len;

    // Check for overflow
    if end_pos < offset || end_pos < len {
        kprint!("sys_fallocate: overflow in offset+len calculation");
        return Err(Err::Inval);
    }

    if !get_p9_enabled() {
        let msg = b"sys_fallocate: p9 is not enabled";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }

    // Validate fd
    if fd >= 256 {
        return Err(Err::BadFd);
    }

    let fd_entry = get_fd(fd);
    if fd_entry.file_desc_id == 0xFF {
        return Err(Err::BadFd);
    }

    let file_desc = get_file_desc(fd_entry.file_desc_id);

    // Check if file descriptor is opened for writing
    // fallocate() requires a file descriptor opened for writing
    const O_RDONLY: u32 = 0;
    const O_WRONLY: u32 = 1;
    const O_RDWR: u32 = 2;
    const O_ACCMODE: u32 = O_RDONLY | O_WRONLY | O_RDWR;

    let accmode = file_desc.flags & O_ACCMODE;
    if accmode == O_RDONLY {
        kprint!("sys_fallocate: fd opened read-only, returning EBADF");
        return Err(Err::BadFd);
    }

    // Validate parameters
    // Check for negative offset or len (cast to signed for comparison)
    let offset_i64 = offset as i64;
    let len_i64 = len as i64;

    if offset_i64 < 0 {
        kprint!(
            "sys_fallocate: negative offset {}, returning EINVAL",
            offset
        );
        return Err(Err::Inval);
    }

    if len_i64 <= 0 {
        kprint!("sys_fallocate: invalid len {}, returning EINVAL", len);
        return Err(Err::Inval);
    }

    // Check for overflow
    if end_pos < offset || end_pos < len {
        kprint!("sys_fallocate: overflow in offset+len calculation");
        return Err(Err::Inval);
    }

    // Validate mode flags
    // FALLOC_FL_KEEP_SIZE = 0x01 (don't change file size, just preallocate space)
    // FALLOC_FL_PUNCH_HOLE = 0x02 (punch a hole)
    // FALLOC_FL_NO_HIDE_STALE = 0x04
    const FALLOC_FL_KEEP_SIZE: u32 = 0x01;

    // For now, we only support the default mode (0) and FALLOC_FL_KEEP_SIZE
    if (mode & !FALLOC_FL_KEEP_SIZE) != 0 {
        kprint!("sys_fallocate: unsupported mode flags 0x{:x}", mode);
        // Return success for unsupported flags (some filesystems don't support all flags)
        // but we'll still try to allocate space
    }

    kprint!(
        "sys_fallocate: fd={}, mode=0x{:x}, offset={}, len={}, end_pos={}",
        fd,
        mode,
        offset,
        len,
        end_pos
    );

    // Get current file size - we always need it
    let tgetattr = crate::p9::TgetattrMessage::new(0, file_desc.fid, P9_GETATTR_SIZE);
    let current_size = match tgetattr.send_tgetattr() {
        Ok(_) => match RgetattrMessage::read_response() {
            P9Response::Success(rgetattr) => rgetattr.size,
            P9Response::Error(_) => {
                kprint!("sys_fallocate: error getting current file size");
                return Err(Err::NoSys);
            }
        },
        Err(_) => {
            kprint!("sys_fallocate: error sending getattr");
            return Err(Err::NoSys);
        }
    };

    kprint!(
        "sys_fallocate: current_size={}, end_pos={}, mode=0x{:x}",
        current_size,
        end_pos,
        mode
    );

    // Handle allocation based on mode
    if (mode & FALLOC_FL_KEEP_SIZE) != 0 {
        // FALLOC_FL_KEEP_SIZE: Preallocate space but don't change file size
        // Since 9P doesn't support sparse allocation, we need to temporarily extend
        // the file to allocate space, then restore the original size
        let size_to_allocate = end_pos.max(current_size);

        // Check if the requested size is too large before attempting allocation
        const MAX_REASONABLE_SIZE: u64 = 0x7FFF_FFFF_FFFF_FFFF; // 2^63 - 1
        if size_to_allocate > MAX_REASONABLE_SIZE || offset > MAX_REASONABLE_SIZE {
            kprint!(
                "sys_fallocate: file size too large: size_to_allocate={}, offset={}",
                size_to_allocate,
                offset
            );
            return Err(Err::FileTooBig);
        }

        if size_to_allocate > current_size {
            // First, extend the file to allocate space
            let valid = P9SetattrMask::Size as u32;
            let tsetattr_extend = TsetattrMessage::new(
                0,
                file_desc.fid,
                valid,
                0,
                0,
                0,
                size_to_allocate,
                0,
                0,
                0,
                0,
            );

            match tsetattr_extend.send_tsetattr() {
                Ok(_) => match RsetattrMessage::read_response() {
                    P9Response::Success(_) => {
                        kprint!(
                            "sys_fallocate: allocated space (extended to {}), now restoring size to {}",
                            size_to_allocate,
                            current_size
                        );

                        // Now restore the original size
                        let tsetattr_restore = TsetattrMessage::new(
                            0,
                            file_desc.fid,
                            valid,
                            0,
                            0,
                            0,
                            current_size,
                            0,
                            0,
                            0,
                            0,
                        );

                        match tsetattr_restore.send_tsetattr() {
                            Ok(_) => match RsetattrMessage::read_response() {
                                P9Response::Success(_) => {
                                    kprint!("sys_fallocate: restored size to {}", current_size);
                                    Ok(0)
                                }
                                P9Response::Error(e) => {
                                    kprint!(
                                        "sys_fallocate: error restoring size: ecode={}",
                                        e.ecode
                                    );
                                    Err(Err::NoSys)
                                }
                            },
                            Err(e) => {
                                kprint!("sys_fallocate: error sending restore Tsetattr: {:?}", e);
                                Err(Err::NoSys)
                            }
                        }
                    }
                    P9Response::Error(rlerror) => {
                        kprint!(
                            "sys_fallocate: error extending file: ecode={}, size_to_allocate={}",
                            rlerror.ecode,
                            size_to_allocate
                        );
                        // Map 9P error codes to Linux error codes
                        if rlerror.ecode == 27 {
                            Err(Err::FileTooBig)
                        } else if rlerror.ecode == 22
                            && (size_to_allocate > 0x7FFF_FFFF_FFFF_FFFF
                                || offset > 0x7FFF_FFFF_FFFF_FFFF)
                        {
                            // If the file size or offset is very large, treat EINVAL as EFBIG
                            Err(Err::FileTooBig)
                        } else {
                            Err(Err::Inval)
                        }
                    }
                },
                Err(e) => {
                    kprint!("sys_fallocate: error sending extend Tsetattr: {:?}", e);
                    Err(Err::NoSys)
                }
            }
        } else {
            // Already large enough, nothing to do
            kprint!(
                "sys_fallocate: file already large enough (size={})",
                current_size
            );
            Ok(0)
        }
    } else {
        // Default mode: extend file to end_pos (or keep current if larger)
        // For fallocate, we should extend the file to at least offset + len
        // even if the file is already larger
        let target_size = end_pos.max(current_size);

        kprint!(
            "sys_fallocate: default mode - current_size={}, end_pos={}, target_size={}",
            current_size,
            end_pos,
            target_size
        );

        if target_size > current_size {
            let valid = P9SetattrMask::Size as u32;
            let tsetattr =
                TsetattrMessage::new(0, file_desc.fid, valid, 0, 0, 0, target_size, 0, 0, 0, 0);

            match tsetattr.send_tsetattr() {
                Ok(_) => match RsetattrMessage::read_response() {
                    P9Response::Success(_) => {
                        kprint!(
                            "sys_fallocate: successfully extended file to {}",
                            target_size
                        );
                        Ok(0)
                    }
                    P9Response::Error(rlerror) => {
                        kprint!("sys_fallocate: error setting size: ecode={}", rlerror.ecode);
                        // Map 9P error codes to Linux error codes
                        // 27 = EFBIG (File too large)
                        // 22 = EINVAL (Invalid argument)
                        // For fallocate, if we're trying to allocate a very large file,
                        // the server might return EINVAL, but we should check if it's actually EFBIG
                        // MAX_FILESIZE is typically LLONG_MAX / 1024, and with 1024-byte blocks
                        // the offset would be around 9 exabytes, which is > 0x7FFF_FFFF_FFFF_FFFF
                        const MAX_REASONABLE_SIZE: u64 = 0x7FFF_FFFF_FFFF_FFFF; // 2^63 - 1
                        if rlerror.ecode == 27 {
                            Err(Err::FileTooBig)
                        } else if rlerror.ecode == 22
                            && (target_size > MAX_REASONABLE_SIZE || offset > MAX_REASONABLE_SIZE)
                        {
                            // If the file size or offset is very large, treat EINVAL as EFBIG
                            Err(Err::FileTooBig)
                        } else {
                            Err(Err::Inval)
                        }
                    }
                },
                Err(e) => {
                    kprint!("sys_fallocate: error sending Tsetattr: {:?}", e);
                    Err(Err::NoSys)
                }
            }
        } else {
            // Already large enough, nothing to do
            kprint!(
                "sys_fallocate: file already large enough (size={} >= target={})",
                current_size,
                target_size
            );
            Ok(0)
        }
    }
}

pub fn sys_fanotify_init(_flags: u32, _event_f_flags: u32) -> Result<u32, Err> {
    let msg = b"sys_fanotify_init not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_fanotify_mark(
    _fanotify_fd: u32,
    _flags: u32,
    _mask: u32,
    _dirfd: u32,
    _pathname: u32,
) -> Result<u32, Err> {
    let msg = b"sys_fanotify_mark not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_fchdir(fd: u32) -> Result<u32, Err> {
    if !get_p9_enabled() {
        let msg = b"sys_fchdir: p9 is not enabled";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }

    // Validate fd range
    if fd >= 256 {
        return Err(Err::BadFd);
    }

    let fd_entry = get_fd(fd);
    if fd_entry.file_desc_id == 0xFF {
        // Invalid file descriptor
        return Err(Err::BadFd);
    }

    let file_desc = get_file_desc(fd_entry.file_desc_id);

    // Check if the file descriptor refers to a directory
    // For now, we'll check this after setting CWD and let the filesystem handle it
    // In a more complete implementation, we should verify it's a directory first
    set_cwd_fid(file_desc.fid);
    Ok(0)
}

pub fn sys_fchmod(fd: u32, mode: u32) -> Result<u32, Err> {
    if !get_p9_enabled() {
        let msg = b"sys_fchmod: p9 is not enabled";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }

    // Validate fd
    if fd >= 256 {
        return Err(Err::BadFd);
    }

    let fd_entry = get_fd(fd);
    if fd_entry.file_desc_id == 0xFF {
        return Err(Err::BadFd);
    }

    let file_desc = get_file_desc(fd_entry.file_desc_id);

    kprint!("sys_fchmod: fd={}, mode=0o{:o}", fd, mode);

    // Use Tsetattr to change file mode (permissions)
    // We only want to set the mode, so we use the Mode valid flag
    let valid = P9SetattrMask::Mode as u32;

    // Extract the permission bits (low 12 bits for mode)
    let p9_mode = mode & 0o7777;

    let tsetattr = TsetattrMessage::new(
        0,             // tag
        file_desc.fid, // fid
        valid,         // valid mask (only Mode)
        p9_mode,       // mode (permissions)
        0,             // uid (not changing)
        0,             // gid (not changing)
        0,             // size (not changing)
        0,             // atime_sec (not changing)
        0,             // atime_nsec (not changing)
        0,             // mtime_sec (not changing)
        0,             // mtime_nsec (not changing)
    );

    match tsetattr.send_tsetattr() {
        Ok(_) => match RsetattrMessage::read_response() {
            P9Response::Success(_) => {
                kprint!("sys_fchmod: successfully changed mode to 0o{:o}", p9_mode);
                Ok(0)
            }
            P9Response::Error(rlerror) => {
                kprint!("sys_fchmod: error setting mode: ecode={}", rlerror.ecode);
                Err(Err::NoSys)
            }
        },
        Err(e) => {
            kprint!("sys_fchmod: error sending Tsetattr: {:?}", e);
            Err(Err::NoSys)
        }
    }
}

pub fn sys_fchmodat(dfd: u32, filename: u32, mode: u32, flag: u32) -> Result<u32, Err> {
    if !get_p9_enabled() {
        let msg = b"sys_fchmodat: p9 is not enabled";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }

    // Validate flag
    // For fchmodat, only AT_SYMLINK_NOFOLLOW (0x100) is valid
    // But we should only reject clearly invalid flags (like -1), not unknown ones
    const AT_SYMLINK_NOFOLLOW: u32 = 0x100;
    let flag_i32 = flag as i32;
    // Only reject negative flags (like -1 = 0xFFFFFFFF)
    // Allow unknown flag values to pass through (they'll be ignored)
    if flag_i32 < 0 {
        kprint!("sys_fchmodat: negative flag 0x{:x}", flag);
        return Err(Err::Inval);
    }

    // Check for known invalid combinations, but allow unknown flags
    // For now, we'll just log if AT_SYMLINK_NOFOLLOW is set
    if (flag & AT_SYMLINK_NOFOLLOW) != 0 {
        kprint!("sys_fchmodat: AT_SYMLINK_NOFOLLOW flag set (not yet implemented)");
    }

    // Check for invalid address
    if filename == 0 {
        kprint!("sys_fchmodat: invalid filename address (NULL)");
        return Err(Err::Fault);
    }

    // Check if address is in user memory range
    if filename >= 0xC0000000 {
        kprint!(
            "sys_fchmodat: invalid filename address (out of user memory): 0x{:x}",
            filename
        );
        return Err(Err::Fault);
    }

    // Read the filename from user space
    let filename_buf = unsafe { core::slice::from_raw_parts(filename as *const u8, 256) };
    let null_pos = filename_buf
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(filename_buf.len());
    let filename_slice = &filename_buf[..null_pos];
    let filename_str = match str::from_utf8(filename_slice) {
        Ok(s) => s,
        Err(_) => {
            kprint!("sys_fchmodat: invalid UTF-8 filename");
            return Err(Err::NoSys);
        }
    };

    // Check for pathname too long (PATH_MAX is typically 4096, but for 9P we limit to 256)
    // Only check if path is non-empty (empty paths may be valid in some contexts)
    if !filename_str.is_empty() && filename_str.len() > 255 {
        kprint!(
            "sys_fchmodat: pathname too long ({} bytes), returning ENAMETOOLONG",
            filename_str.len()
        );
        return Err(Err::NameTooLong);
    }

    // Check for empty path (should return ENOENT for fchmodat)
    // But only if it's not a special case like AT_EMPTY_PATH
    if filename_str.is_empty() {
        // For fchmodat, empty pathname should return ENOENT
        // (unlike fchmod which works on a file descriptor)
        kprint!("sys_fchmodat: empty pathname, returning ENOENT");
        return Err(Err::FileNotFound);
    }

    kprint!(
        "sys_fchmodat: dfd={}, filename='{}', mode={}, flag={}",
        dfd,
        filename_str,
        mode,
        flag
    );

    resolve_path(dfd, filename_str, 0xFFFF_FFFE)?;

    // Create Tsetattr message for chmod operation
    // We only want to set the mode, so we use the Mode valid flag
    let valid = P9SetattrMask::Mode as u32;

    // Extract the permission bits (low 12 bits for mode)
    let p9_mode = mode & 0o7777;

    // For chmod, we don't want to change other attributes, so we use default values
    // and only set the mode
    let tsetattr = TsetattrMessage::new(
        0,           // tag
        0xFFFF_FFFE, // fid (the file we walked to)
        valid,       // valid mask (only Mode)
        p9_mode,     // mode (the new permissions)
        0,           // uid (not changing)
        0,           // gid (not changing)
        0,           // size (not changing)
        0,           // atime_sec (not changing)
        0,           // atime_nsec (not changing)
        0,           // mtime_sec (not changing)
        0,           // mtime_nsec (not changing)
    );

    match tsetattr.send_tsetattr() {
        Ok(bytes_written) => {
            kprint!("sys_fchmodat: sent {} bytes for Tsetattr", bytes_written);
        }
        Err(e) => {
            kprint!("sys_fchmodat: error sending Tsetattr: {:?}", e);
            return Err(Err::NoSys);
        }
    }

    match RsetattrMessage::read_response() {
        P9Response::Success(rsetattr) => {
            kprint!("sys_fchmodat: rsetattr = {:?}", rsetattr);
            Ok(0)
        }
        P9Response::Error(rlerror) => {
            kprint!(
                "sys_fchmodat: received Rlerror for Rsetattr: tag={}, ecode={}",
                rlerror.tag,
                rlerror.ecode
            );
            Ok(-(rlerror.ecode as i32) as u32)
        }
    }
}

pub fn sys_fchmodat2(dfd: u32, filename: u32, mode: u32, flag: u32) -> Result<u32, Err> {
    // fchmodat2 is essentially the same as fchmodat, just delegate to it
    sys_fchmodat(dfd, filename, mode, flag)
}

pub fn sys_fchown(fd: u32, user: u32, group: u32) -> Result<u32, Err> {
    if !get_p9_enabled() {
        let msg = b"sys_fchown: p9 is not enabled";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }

    // Validate fd
    if fd >= 256 {
        return Err(Err::BadFd);
    }

    let fd_entry = get_fd(fd);
    if fd_entry.file_desc_id == 0xFF {
        return Err(Err::BadFd);
    }

    let file_desc = get_file_desc(fd_entry.file_desc_id);

    kprint!("sys_fchown: fd={}, uid={}, gid={}", fd, user, group);

    // Use Tsetattr to change file ownership
    // We want to set both uid and gid, so we use both User and Group valid flags
    let valid = P9SetattrMask::Uid as u32 | P9SetattrMask::Gid as u32;

    let tsetattr = TsetattrMessage::new(
        0,             // tag
        file_desc.fid, // fid
        valid,         // valid mask (Uid and Gid)
        0,             // mode (not changing)
        user,          // uid (the new user ID)
        group,         // gid (the new group ID)
        0,             // size (not changing)
        0,             // atime_sec (not changing)
        0,             // atime_nsec (not changing)
        0,             // mtime_sec (not changing)
        0,             // mtime_nsec (not changing)
    );

    match tsetattr.send_tsetattr() {
        Ok(_) => match RsetattrMessage::read_response() {
            P9Response::Success(_) => {
                kprint!(
                    "sys_fchown: successfully changed ownership to uid={}, gid={}",
                    user,
                    group
                );
                Ok(0)
            }
            P9Response::Error(rlerror) => {
                kprint!(
                    "sys_fchown: error setting ownership: ecode={}",
                    rlerror.ecode
                );
                Err(Err::NoSys)
            }
        },
        Err(e) => {
            kprint!("sys_fchown: error sending Tsetattr: {:?}", e);
            Err(Err::NoSys)
        }
    }
}

pub fn sys_fchownat(dfd: u32, filename: u32, user: u32, group: u32, flag: u32) -> Result<u32, Err> {
    if !get_p9_enabled() {
        let msg = b"sys_fchownat: p9 is not enabled";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }

    // Validate flag (AT_SYMLINK_NOFOLLOW and AT_EMPTY_PATH are valid flags)
    const AT_SYMLINK_NOFOLLOW: u32 = 0x100;
    const AT_EMPTY_PATH: u32 = 0x1000;
    const AT_FLAGS_MASK: u32 = AT_SYMLINK_NOFOLLOW | AT_EMPTY_PATH;

    // Only reject negative flags (like -1 = 0xFFFFFFFF)
    let flag_i32 = flag as i32;
    if flag_i32 < 0 || (flag & !AT_FLAGS_MASK) != 0 {
        kprint!("sys_fchownat: invalid flag 0x{:x}", flag);
        return Err(Err::Inval);
    }

    // Read the filename from user space
    let filename_buf = unsafe { core::slice::from_raw_parts(filename as *const u8, 256) };
    let null_pos = filename_buf
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(filename_buf.len());
    let filename_slice = &filename_buf[..null_pos];
    let filename_str = match str::from_utf8(filename_slice) {
        Ok(s) => s,
        Err(_) => {
            kprint!("sys_fchownat: invalid UTF-8 filename");
            return Err(Err::NoSys);
        }
    };

    kprint!(
        "sys_fchownat: dfd={}, filename='{}', user={}, group={}, flag={}",
        dfd,
        filename_str,
        user,
        group,
        flag
    );

    // Handle AT_SYMLINK_NOFOLLOW: don't follow symlinks, operate on the symlink itself
    if (flag & AT_SYMLINK_NOFOLLOW) != 0 {
        kprint!("sys_fchownat: AT_SYMLINK_NOFOLLOW flag set, operating on symlink directly");

        // Walk to the file without following symlinks
        let starting_fid = get_starting_fid(dfd, filename_str)?;
        let (dir_path, file_name) = split_path(filename_str);
        kprint!(
            "sys_fchownat: dir_path='{}', file_name='{}'",
            dir_path,
            file_name
        );

        // For relative paths, use the dfd directly as the directory
        let dir_fid = if dir_path.is_empty() || dir_path == "." {
            // Relative path - use dfd as the directory
            starting_fid
        } else {
            // Absolute path - walk to the directory first
            let dir_path = normalize_path(&dir_path);
            if do_walk(starting_fid, 0xFFFF_FFFB, dir_path).is_err() {
                clunk(0xFFFF_FFFB, false);
                return Err(Err::FileNotFound);
            }
            0xFFFF_FFFB
        };

        // Now walk to the final component (the symlink itself) without following
        // We use Twalk directly on just the final component to avoid symlink resolution
        let file_path = vec![file_name.clone()];
        kprint!(
            "sys_fchownat: walking to symlink: dir_fid={}, file_name='{}', target_fid=0xFFFF_FFFE",
            dir_fid,
            file_name
        );
        let twalk = crate::p9::TwalkMessage::new(0, dir_fid, 0xFFFF_FFFE, file_path.clone());
        match twalk.send_twalk() {
            Ok(_) => match RwalkMessage::read_response() {
                P9Response::Success(rwalk) => {
                    if rwalk.wqids.len() != 1 {
                        clunk(0xFFFF_FFFE, false);
                        clunk(0xFFFF_FFFB, false);
                        return Err(Err::FileNotFound);
                    }

                    // Check if the final component is a symlink
                    if let Some(last_qid) = rwalk.wqids.last() {
                        if last_qid.is_symlink() {
                            kprint!(
                                "sys_fchownat: symlink detected with AT_SYMLINK_NOFOLLOW, operating on symlink directly"
                            );
                            // We have the symlink FID at 0xFFFF_FFFE, don't follow it
                        } else {
                            kprint!("sys_fchownat: not a symlink, operating normally");
                            // Not a symlink, continue normally
                        }
                    }
                }
                P9Response::Error(rlerror) => {
                    kprint!(
                        "sys_fchownat: error walking to file: ecode={}",
                        rlerror.ecode
                    );
                    if dir_path.is_empty() || dir_path == "." {
                        // No cleanup needed for relative paths
                    } else {
                        clunk(0xFFFF_FFFB, false);
                    }
                    return Err(Err::FileNotFound);
                }
            },
            Err(e) => {
                kprint!("sys_fchownat: error sending Twalk: {:?}", e);
                if !dir_path.is_empty() && dir_path != "." {
                    clunk(0xFFFF_FFFB, false);
                }
                return Err(Err::NoSys);
            }
        }
    } else {
        // Normal path resolution (follows symlinks)
        resolve_path(dfd, filename_str, 0xFFFF_FFFE)?;
    }

    // Create Tsetattr message for chown operation
    // We only want to set UID and GID, so we use the appropriate valid flags
    let valid = P9SetattrMask::Uid as u32 | P9SetattrMask::Gid as u32;

    kprint!(
        "sys_fchownat: calling Tsetattr on FID 0xFFFF_FFFE with user={}, group={}",
        user,
        group
    );

    // For chown, we don't want to change other attributes, so we use default values
    // and only set the UID and GID
    let tsetattr = TsetattrMessage::new(
        0,           // tag
        0xFFFF_FFFE, // fid (the file we walked to)
        valid,       // valid mask (only UID and GID)
        0,           // mode (not changing)
        user,        // uid
        group,       // gid
        0,           // size (not changing)
        0,           // atime_sec (not changing)
        0,           // atime_nsec (not changing)
        0,           // mtime_sec (not changing)
        0,           // mtime_nsec (not changing)
    );

    match tsetattr.send_tsetattr() {
        Ok(bytes_written) => {
            kprint!("sys_fchownat: sent {} bytes for Tsetattr", bytes_written);
        }
        Err(e) => {
            kprint!("sys_fchownat: error sending Tsetattr: {:?}", e);
            return Err(Err::NoSys);
        }
    }

    match RsetattrMessage::read_response() {
        P9Response::Success(rsetattr) => {
            kprint!("sys_fchownat: rsetattr = {:?}", rsetattr);

            // Clean up temporary FIDs if we used AT_SYMLINK_NOFOLLOW
            if (flag & AT_SYMLINK_NOFOLLOW) != 0 {
                let (dir_path, _) = split_path(filename_str);
                if !dir_path.is_empty() && dir_path != "." {
                    clunk(0xFFFF_FFFB, false);
                }
            }

            Ok(0)
        }
        P9Response::Error(rlerror) => {
            kprint!(
                "sys_fchownat: received Rlerror for Rsetattr: tag={}, ecode={}",
                rlerror.tag,
                rlerror.ecode
            );

            // Clean up temporary FIDs if we used AT_SYMLINK_NOFOLLOW
            if (flag & AT_SYMLINK_NOFOLLOW) != 0 {
                let (dir_path, _) = split_path(filename_str);
                if !dir_path.is_empty() && dir_path != "." {
                    clunk(0xFFFF_FFFB, false);
                }
            }

            Ok(-(rlerror.ecode as i32) as u32)
        }
    }
}

pub fn sys_utimensat_time64(dfd: u32, filename: u32, times: u32, flags: u32) -> Result<u32, Err> {
    if !get_p9_enabled() {
        let msg = b"sys_utimensat_time64: p9 is not enabled";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }

    // Parse filename
    let filename_buf = unsafe { core::slice::from_raw_parts(filename as *const u8, 256) };
    let null_pos = filename_buf
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(filename_buf.len());
    let filename_slice = &filename_buf[..null_pos];
    let filename_str = match str::from_utf8(filename_slice) {
        Ok(s) => s,
        Err(_) => {
            kprint!("sys_utimensat_time64: invalid UTF-8 filename");
            return Err(Err::NoSys);
        }
    };

    kprint!(
        "sys_utimensat_time64: dfd={}, filename='{}', times={}, flags={}",
        dfd,
        filename_str,
        times,
        flags
    );

    // Handle AT_EMPTY_PATH - update times on the fd itself
    const AT_EMPTY_PATH: u32 = 0x1000;
    if (flags & AT_EMPTY_PATH) != 0 && filename_str.is_empty() {
        // Update times on the file descriptor itself
        if dfd >= 256 {
            return Err(Err::BadFd);
        }

        let fd_entry = get_fd(dfd);
        if fd_entry.file_desc_id == 0xFF {
            return Err(Err::BadFd);
        }

        let file_desc = get_file_desc(fd_entry.file_desc_id);
        let fid = file_desc.fid;

        // Parse times and update
        let (atime_sec, atime_nsec, mtime_sec, mtime_nsec) = parse_times(times)?;

        let valid = P9SetattrMask::Atime as u32 | P9SetattrMask::Mtime as u32;
        let tsetattr = TsetattrMessage::new(
            0,          // tag
            fid,        // fid
            valid,      // valid mask (Atime and Mtime)
            0,          // mode (not changing)
            0,          // uid (not changing)
            0,          // gid (not changing)
            0,          // size (not changing)
            atime_sec,  // atime_sec
            atime_nsec, // atime_nsec
            mtime_sec,  // mtime_sec
            mtime_nsec, // mtime_nsec
        );

        match tsetattr.send_tsetattr() {
            Ok(_) => match RsetattrMessage::read_response() {
                P9Response::Success(_) => Ok(0),
                P9Response::Error(_) => Err(Err::NoSys),
            },
            Err(_) => Err(Err::NoSys),
        }
    } else {
        // Normal path-based update
        resolve_path(dfd, filename_str, 0xFFFF_FFFE)?;

        // Parse times
        let (atime_sec, atime_nsec, mtime_sec, mtime_nsec) = parse_times(times)?;

        // Create Tsetattr message to update times
        let valid = P9SetattrMask::Atime as u32 | P9SetattrMask::Mtime as u32;
        let tsetattr = TsetattrMessage::new(
            0,           // tag
            0xFFFF_FFFE, // fid (the file we walked to)
            valid,       // valid mask (Atime and Mtime)
            0,           // mode (not changing)
            0,           // uid (not changing)
            0,           // gid (not changing)
            0,           // size (not changing)
            atime_sec,   // atime_sec
            atime_nsec,  // atime_nsec
            mtime_sec,   // mtime_sec
            mtime_nsec,  // mtime_nsec
        );

        match tsetattr.send_tsetattr() {
            Ok(_) => match RsetattrMessage::read_response() {
                P9Response::Success(_) => {
                    clunk(0xFFFF_FFFE, false);
                    Ok(0)
                }
                P9Response::Error(rlerror) => {
                    clunk(0xFFFF_FFFE, false);
                    kprint!("sys_utimensat_time64: Rlerror: ecode={}", rlerror.ecode);
                    Err(Err::NoSys)
                }
            },
            Err(e) => {
                clunk(0xFFFF_FFFE, false);
                kprint!("sys_utimensat_time64: error: {:?}", e);
                Err(Err::NoSys)
            }
        }
    }
}

// Helper function to parse times array
fn parse_times(times: u32) -> Result<(u64, u64, u64, u64), Err> {
    const UTIME_OMIT: i64 = 0x3fffffff;
    const UTIME_NOW: i64 = 0x3ffffffe;

    if times == 0 {
        // NULL times - update both to current time (0 means "now" in 9P)
        return Ok((0, 0, 0, 0));
    }

    // Parse the timespec array: struct timespec64[2] = {atime, mtime}
    // timespec64: {tv_sec: i64, tv_nsec: i64}
    let timespec_array = unsafe { core::slice::from_raw_parts(times as *const i64, 4) };

    let atime_sec_raw = timespec_array[0];
    let atime_nsec_raw = timespec_array[1];

    let mtime_sec_raw = timespec_array[2];
    let mtime_nsec_raw = timespec_array[3];

    // Handle UTIME_OMIT and UTIME_NOW
    let atime_sec = if atime_sec_raw == UTIME_OMIT || atime_nsec_raw == UTIME_OMIT {
        return Err(Err::NoSys); // UTIME_OMIT requires preserving current time
    } else if atime_sec_raw == UTIME_NOW || atime_nsec_raw == UTIME_NOW {
        0 // 0 means "now" in 9P
    } else {
        atime_sec_raw as u64
    };

    let atime_nsec = if atime_sec_raw == UTIME_NOW || atime_nsec_raw == UTIME_NOW {
        0
    } else {
        (atime_nsec_raw as u64) & 0xFFFFFFFF // Keep only 32 bits for nsec
    };

    let mtime_sec = if mtime_sec_raw == UTIME_OMIT || mtime_nsec_raw == UTIME_OMIT {
        return Err(Err::NoSys);
    } else if mtime_sec_raw == UTIME_NOW || mtime_nsec_raw == UTIME_NOW {
        0
    } else {
        mtime_sec_raw as u64
    };

    let mtime_nsec = if mtime_sec_raw == UTIME_NOW || mtime_nsec_raw == UTIME_NOW {
        0
    } else {
        (mtime_nsec_raw as u64) & 0xFFFFFFFF
    };

    Ok((atime_sec, atime_nsec, mtime_sec, mtime_nsec))
}

pub fn sys_fdatasync(_fd: u32) -> Result<u32, Err> {
    let msg = b"sys_fdatasync not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_fgetxattr(fd: u32, name: u32, value: u32, size: u32) -> Result<u32, Err> {
    if !get_p9_enabled() {
        return Err(Err::NoSys);
    }

    // Validate fd
    if fd >= 256 {
        return Err(Err::BadFd);
    }

    let fd_entry = get_fd(fd);
    if fd_entry.file_desc_id == 0xFF {
        return Err(Err::BadFd);
    }

    let file_desc = get_file_desc(fd_entry.file_desc_id);

    // Parse the xattr name
    let name_buf = unsafe { core::slice::from_raw_parts(name as *const u8, 256) };
    let name_null_pos = name_buf
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(name_buf.len());
    let name_str = match str::from_utf8(&name_buf[..name_null_pos]) {
        Ok(s) => s,
        Err(_) => return Err(Err::Inval),
    };

    kprint!(
        "sys_fgetxattr: fd={} name='{}' value=0x{:x} size={}",
        fd,
        name_str,
        value,
        size
    );

    // Use Txattrwalk to get the xattr
    let txattrwalk =
        crate::p9::TxattrwalkMessage::new(0, file_desc.fid, 0xFFFF_FFF9, name_str.to_string());

    match txattrwalk.send_txattrwalk() {
        Ok(_) => {}
        Err(_) => return Err(Err::NoSys),
    }

    match crate::p9::RxattrwalkMessage::read_response() {
        crate::p9::P9Response::Success(rxattrwalk) => {
            let xattr_size = rxattrwalk.size;
            kprint!("sys_fgetxattr: xattr size = {}", xattr_size);

            // If size is 0, just return the size
            if size == 0 {
                clunk(0xFFFF_FFF9, false);
                return Ok(xattr_size as u32);
            }

            // If buffer is too small, return the required size
            if size < xattr_size as u32 {
                clunk(0xFFFF_FFF9, false);
                return Err(Err::Range); // E2BIG
            }

            // Read the xattr value
            let tread = crate::p9::TreadMessage::new(0, 0xFFFF_FFF9, 0, xattr_size as u32);
            match tread.send_tread() {
                Ok(_) => {}
                Err(_) => {
                    clunk(0xFFFF_FFF9, false);
                    return Err(Err::NoSys);
                }
            }

            match crate::p9::RreadMessage::read_response() {
                crate::p9::P9Response::Success(rread) => {
                    let data_len = rread.data.len().min(size as usize);
                    if value != 0 {
                        unsafe {
                            let dest = core::slice::from_raw_parts_mut(value as *mut u8, data_len);
                            dest.copy_from_slice(&rread.data[..data_len]);
                        }
                    }
                    clunk(0xFFFF_FFF9, false);
                    Ok(data_len as u32)
                }
                crate::p9::P9Response::Error(_) => {
                    clunk(0xFFFF_FFF9, false);
                    Err(Err::NoSys)
                }
            }
        }
        crate::p9::P9Response::Error(rlerror) => {
            kprint!("sys_fgetxattr: xattrwalk failed: ecode={}", rlerror.ecode);
            Err(Err::NoData) // ENODATA - no such attribute
        }
    }
}

pub fn sys_file_getattr(_dfd: u32, _filename: u32, _mask: u32) -> Result<u32, Err> {
    let msg = b"sys_file_getattr not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_file_setattr(_dfd: u32, _filename: u32, _mask: u32) -> Result<u32, Err> {
    let msg = b"sys_file_setattr not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_flistxattr(_fd: u32, _list: u32, _size: u32) -> Result<u32, Err> {
    let msg = b"sys_flistxattr not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_flock(_fd: u32, _cmd: u32) -> Result<u32, Err> {
    let msg = b"sys_flock not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_fremovexattr(_fd: u32, _name: u32) -> Result<u32, Err> {
    let msg = b"sys_fremovexattr not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_ftruncate64(fd: u32, length: u32) -> Result<u32, Err> {
    if !get_p9_enabled() {
        let msg = b"sys_ftruncate64: p9 is not enabled";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }

    if fd >= 256 {
        kprint!("sys_ftruncate64: invalid file descriptor {}", fd);
        return Err(Err::Inval);
    }

    kprint!("sys_ftruncate64: fd={}, length={}", fd, length);

    // Get the file descriptor entry
    let fd_entry = get_fd(fd);
    if fd_entry.file_desc_id == 0xFF {
        kprint!("sys_ftruncate64: invalid file descriptor {}", fd);
        return Err(Err::BadFd);
    }

    let file_desc = get_file_desc(fd_entry.file_desc_id);

    // Create Tsetattr message for truncate operation
    // We only want to set the size, so we use the Size valid flag
    let valid = P9SetattrMask::Size as u32;
    let new_size = length as u64;

    // For ftruncate, we don't want to change other attributes, so we use default values
    // and only set the size
    let tsetattr = TsetattrMessage::new(
        0,             // tag
        file_desc.fid, // fid (the file we want to truncate)
        valid,         // valid mask (only Size)
        0,             // mode (not changing)
        0,             // uid (not changing)
        0,             // gid (not changing)
        new_size,      // size (the new file size)
        0,             // atime_sec (not changing)
        0,             // atime_nsec (not changing)
        0,             // mtime_sec (not changing)
        0,             // mtime_nsec (not changing)
    );

    match tsetattr.send_tsetattr() {
        Ok(bytes_written) => {
            kprint!("sys_ftruncate64: sent {} bytes for Tsetattr", bytes_written);
        }
        Err(e) => {
            kprint!("sys_ftruncate64: error sending Tsetattr: {:?}", e);
            return Err(Err::NoSys);
        }
    }

    match RsetattrMessage::read_response() {
        P9Response::Success(rsetattr) => {
            kprint!("sys_ftruncate64: rsetattr = {:?}", rsetattr);
            Ok(0)
        }
        P9Response::Error(rlerror) => {
            kprint!(
                "sys_ftruncate64: received Rlerror for Rsetattr: tag={}, ecode={}",
                rlerror.tag,
                rlerror.ecode
            );
            Ok(-(rlerror.ecode as i32) as u32)
        }
    }
}

pub fn sys_fsync(_fd: u32) -> Result<u32, Err> {
    let msg = b"sys_fsync not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_fsetxattr(fd: u32, name: u32, value: u32, size: u32, flags: u32) -> Result<u32, Err> {
    if !get_p9_enabled() {
        return Err(Err::NoSys);
    }

    // Validate fd
    if fd >= 256 {
        return Err(Err::BadFd);
    }

    let fd_entry = get_fd(fd);
    if fd_entry.file_desc_id == 0xFF {
        return Err(Err::BadFd);
    }

    let file_desc = get_file_desc(fd_entry.file_desc_id);

    // Parse the xattr name
    let name_buf = unsafe { core::slice::from_raw_parts(name as *const u8, 256) };
    let name_null_pos = name_buf
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(name_buf.len());
    let name_str = match str::from_utf8(&name_buf[..name_null_pos]) {
        Ok(s) => s,
        Err(_) => return Err(Err::Inval),
    };

    kprint!(
        "sys_fsetxattr: fd={} name='{}' value=0x{:x} size={} flags={}",
        fd,
        name_str,
        value,
        size,
        flags
    );

    // Use Txattrcreate to prepare to set the xattr
    // We need to allocate a new FID for the xattr operation
    // The FID passed to Txattrcreate will be used for writing the xattr value
    let xattr_fid = 0xFFFF_FFF8; // Use a temporary FID for the xattr
    
    // First, walk to duplicate the file's FID to the xattr FID
    let twalk = crate::p9::TwalkMessage::new(0, file_desc.fid, xattr_fid, vec![]);
    match twalk.send_twalk() {
        Ok(_) => {}
        Err(_) => return Err(Err::NoSys),
    }
    
    match crate::p9::RwalkMessage::read_response() {
        crate::p9::P9Response::Success(_) => {
            kprint!("sys_fsetxattr: walked to duplicate fid={}", xattr_fid);
        }
        crate::p9::P9Response::Error(rlerror) => {
            kprint!("sys_fsetxattr: walk failed: ecode={}", rlerror.ecode);
            return Err(Err::NoSys);
        }
    }
    
    // Now use Txattrcreate with the xattr FID
    let txattrcreate = crate::p9::TxattrcreateMessage::new(
        0,
        xattr_fid, // The FID that will be used for the xattr
        name_str.to_string(),
        size as u64,
        flags,
    );

    match txattrcreate.send_txattrcreate() {
        Ok(_) => {}
        Err(_) => {
            clunk(xattr_fid, false);
            return Err(Err::NoSys);
        }
    }

    match crate::p9::RxattrcreateMessage::read_response() {
        crate::p9::P9Response::Success(_) => {
            kprint!("sys_fsetxattr: xattrcreate succeeded, using fid={}", xattr_fid);

            // Now write the xattr value to the xattr FID
            if size > 0 && value != 0 {
                let value_buf =
                    unsafe { core::slice::from_raw_parts(value as *const u8, size as usize) };

                // Write to the xattr FID
                let twrite = crate::p9::TwriteMessage::new(0, xattr_fid, 0, value_buf.to_vec());
                match twrite.send_twrite() {
                    Ok(_) => {}
                    Err(_) => {
                        clunk(xattr_fid, false);
                        return Err(Err::NoSys);
                    }
                }

                match crate::p9::RwriteMessage::read_response() {
                    crate::p9::P9Response::Success(rwrite) => {
                        kprint!("sys_fsetxattr: wrote {} bytes", rwrite.count);
                        if rwrite.count != size {
                            kprint!(
                                "sys_fsetxattr: write size mismatch: expected {}, got {}",
                                size,
                                rwrite.count
                            );
                            clunk(xattr_fid, false);
                            return Err(Err::IO);
                        }
                    }
                    crate::p9::P9Response::Error(_) => {
                        clunk(xattr_fid, false);
                        return Err(Err::NoSys);
                    }
                }
            }

            // Clunk the xattr FID to commit the xattr
            clunk(xattr_fid, false);
            Ok(0)
        }
        crate::p9::P9Response::Error(rlerror) => {
            kprint!("sys_fsetxattr: xattrcreate failed: ecode={}", rlerror.ecode);
            Err(Err::NoSys)
        }
    }
}

pub fn sys_getcwd(_buf: u32, _size: u32) -> Result<u32, Err> {
    if _buf == 0 {
        let msg = b"sys_getcwd: _buf is null";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }
    let mut cwd_str = get_cwd_str();
    // strip the trailing / if it's not "/"
    if cwd_str != "/" && cwd_str.ends_with("/") {
        cwd_str = cwd_str[..cwd_str.len() - 1].to_string();
    }
    // add a null-terminator
    cwd_str.push('\0');
    let cwd_str_len = cwd_str.len();
    if cwd_str_len > _size as usize {
        return Err(Err::NoSys);
    }
    let buf = unsafe { core::slice::from_raw_parts_mut(_buf as *mut u8, cwd_str_len) };
    // copy the string into buffer, null-terminated utf-8
    buf.copy_from_slice(cwd_str.as_bytes());
    kprint!(
        "sys_getcwd: cwd_str='{:?}' cwd_str_len={}",
        &buf[..cwd_str_len],
        cwd_str_len
    );
    Ok(cwd_str_len as u32)
}

pub fn sys_getxattr(_pathname: u32, _name: u32, _value: u32, _size: u32) -> Result<u32, Err> {
    let msg = b"sys_getxattr not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_getxattrat(_dfd: u32, _filename: u32, _name: u32, _value: u32) -> Result<u32, Err> {
    let msg = b"sys_getxattrat not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_linkat(
    _olddirfd: u32,
    _oldpath: u32,
    _newdirfd: u32,
    _newpath: u32,
    _flags: u32,
) -> Result<u32, Err> {
    let msg = b"sys_linkat not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_symlinkat(target: u32, newdirfd: u32, linkpath: u32) -> Result<u32, Err> {
    if !get_p9_enabled() {
        let msg = b"sys_symlinkat: p9 is not enabled";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }

    // Parse target (symlink contents) and linkpath (symlink name)
    let target_buf = unsafe { core::slice::from_raw_parts(target as *const u8, 256) };
    let target_null_pos = target_buf
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(target_buf.len());
    let target_str = match str::from_utf8(&target_buf[..target_null_pos]) {
        Ok(s) => s,
        Err(_) => {
            kprint!("sys_symlinkat: invalid UTF-8 target");
            return Err(Err::NoSys);
        }
    };

    let linkpath_buf = unsafe { core::slice::from_raw_parts(linkpath as *const u8, 256) };
    let linkpath_null_pos = linkpath_buf
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(linkpath_buf.len());
    let linkpath_str = match str::from_utf8(&linkpath_buf[..linkpath_null_pos]) {
        Ok(s) => s,
        Err(_) => {
            kprint!("sys_symlinkat: invalid UTF-8 linkpath");
            return Err(Err::NoSys);
        }
    };

    kprint!(
        "sys_symlinkat: target='{}', newdirfd={}, linkpath='{}'",
        target_str,
        newdirfd,
        linkpath_str
    );

    // Split linkpath into directory and filename
    let (dir_path, file_name) = split_path(linkpath_str);
    kprint!(
        "sys_symlinkat: dir_path='{}', file_name='{}'",
        dir_path,
        file_name
    );

    // Resolve the directory where the symlink will be created
    resolve_path(newdirfd, &dir_path, 0xFFFF_FFFE)?;

    // Create symlink using Tsymlink
    let tsymlink = TsymlinkMessage::new(
        0,                      // tag
        0xFFFF_FFFE,            // fid (directory)
        file_name.to_string(),  // name (symlink name)
        target_str.to_string(), // symtgt (symlink target)
        0,                      // gid (default)
    );

    match tsymlink.send_tsymlink() {
        Ok(bytes_written) => {
            kprint!("sys_symlinkat: sent {} bytes for Tsymlink", bytes_written);
        }
        Err(e) => {
            kprint!("sys_symlinkat: error sending Tsymlink: {:?}", e);
            clunk(0xFFFF_FFFE, false);
            return Err(Err::NoSys);
        }
    }

    match RsymlinkMessage::read_response() {
        P9Response::Success(rsymlink) => {
            kprint!("sys_symlinkat: rsymlink = {:?}", rsymlink);
            clunk(0xFFFF_FFFE, false);
            Ok(0)
        }
        P9Response::Error(rlerror) => {
            kprint!(
                "sys_symlinkat: received Rlerror: tag={}, ecode={}",
                rlerror.tag,
                rlerror.ecode
            );
            clunk(0xFFFF_FFFE, false);
            Err(Err::NoSys)
        }
    }
}

pub fn sys_listxattr(_pathname: u32, _list: u32, _size: u32) -> Result<u32, Err> {
    let msg = b"sys_listxattr not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_listxattrat(_dfd: u32, _filename: u32, _list: u32, _size: u32) -> Result<u32, Err> {
    let msg = b"sys_listxattrat not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_llistxattr(_pathname: u32, _list: u32, _size: u32) -> Result<u32, Err> {
    let msg = b"sys_llistxattr not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_llseek(
    _fd: u32,
    _offset_high: u32,
    _offset_low: u32,
    result: u32,
    whence: u32,
) -> Result<u32, Err> {
    const SEEK_SET: u32 = 0;
    const SEEK_CUR: u32 = 1;
    const SEEK_END: u32 = 2;

    if !get_p9_enabled() {
        let msg = b"sys_llseek not implemented";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }

    if _fd >= 256 {
        return Err(Err::Inval);
    }

    if whence != SEEK_SET && whence != SEEK_CUR && whence != SEEK_END {
        let msg = b"sys_llseek: invalid whence";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }

    let fd_entry = get_fd(_fd);
    if fd_entry.file_desc_id == 0xFF {
        return Err(Err::BadFd);
    }

    let file_desc = get_file_desc(fd_entry.file_desc_id);
    let offset = ((_offset_high as u64) << 32) | (_offset_low as u64);
    let new_cursor = match whence {
        SEEK_SET => offset,
        SEEK_CUR => file_desc.cursor + offset,
        SEEK_END => {
            // For SEEK_END, we need to get the file size
            if file_desc.fid != 0 && file_desc.fid != 0xFFFF_FFFE {
                let tgetattr = crate::p9::TgetattrMessage::new(0, file_desc.fid, P9_GETATTR_SIZE);
                match tgetattr.send_tgetattr() {
                    Ok(_) => match RgetattrMessage::read_response() {
                        P9Response::Success(rgetattr) => rgetattr.size + offset,
                        P9Response::Error(_) => {
                            // Fallback: use cursor if getattr fails
                            file_desc.cursor + offset
                        }
                    },
                    Err(_) => {
                        // Fallback: use cursor if getattr fails
                        file_desc.cursor + offset
                    }
                }
            } else {
                file_desc.cursor + offset
            }
        }
        _ => {
            let msg = b"sys_llseek: invalid whence";
            host_log(msg.as_ptr(), msg.len());
            return Err(Err::Inval);
        }
    };

    // Update the shared cursor in the file description
    let mut updated_desc = file_desc;
    updated_desc.cursor = new_cursor;
    update_file_desc(fd_entry.file_desc_id, updated_desc);

    let result_ptr = result as *mut u64;
    unsafe {
        *result_ptr = new_cursor;
    }
    Ok(0)
}

pub fn sys_lremovexattr(_pathname: u32, _name: u32) -> Result<u32, Err> {
    let msg = b"sys_lremovexattr not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_lsetxattr(
    _pathname: u32,
    _name: u32,
    _value: u32,
    _size: u32,
    _flags: u32,
) -> Result<u32, Err> {
    let msg = b"sys_lsetxattr not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_mkdirat(_dfd: u32, _pathname: u32, _mode: u32) -> Result<u32, Err> {
    if !get_p9_enabled() {
        let msg = b"sys_mkdirat: p9 is not enabled";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }
    let filename = unsafe { core::slice::from_raw_parts(_pathname as *const u8, 256) };
    let null_pos = filename
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(filename.len());
    let filename = &filename[..null_pos];
    let filename_str = match str::from_utf8(filename) {
        Ok(s) => s,
        Err(_) => {
            kprint!("sys_mkdirat: invalid UTF-8 filename");
            return Err(Err::NoSys);
        }
    };

    kprint!(
        "sys_mkdirat: dfd={}, filename='{}'",
        _dfd as i32,
        filename_str
    );
    let (dir_path, file_name) = split_path(filename_str);
    kprint!(
        "sys_mkdirat: dir_path='{}', file_name='{}'",
        dir_path,
        file_name
    );
    resolve_path(_dfd, &dir_path, 0xFFFF_FFFE)?;

    let tmkdir = TmkdirMessage::new(0, 0xFFFF_FFFE, file_name.to_string(), _mode, 0);
    match tmkdir.send_tmkdir() {
        Ok(bytes_written) => {
            kprint!("sys_mkdirat: sent {} bytes for Tmkdir", bytes_written);
        }
        Err(e) => {
            kprint!("sys_mkdirat: error sending Tmkdir: {:?}", e);
            return Err(Err::NoSys);
        }
    }
    match RmkdirMessage::read_response() {
        P9Response::Success(rmkdir) => {
            kprint!("sys_mkdirat: rmkdir = {:?}", rmkdir);
            clunk(0xFFFF_FFFE, false);
            Ok(0)
        }
        P9Response::Error(rlerror) => {
            kprint!(
                "sys_mkdirat: received Rlerror for Rmkdir: tag={}, ecode={}",
                rlerror.tag,
                rlerror.ecode
            );
            Ok(-(rlerror.ecode as i32) as u32)
        }
    }
}

pub fn sys_mknodat(_dfd: u32, _filename: u32, _mode: u32, _dev: u32) -> Result<u32, Err> {
    if !get_p9_enabled() {
        let msg = b"sys_mknodat: p9 is not enabled";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }
    let filename = unsafe { core::slice::from_raw_parts(_filename as *const u8, 256) };
    let null_pos = filename
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(filename.len());
    let filename = &filename[..null_pos];
    let filename_str = match str::from_utf8(filename) {
        Ok(s) => s,
        Err(_) => {
            kprint!("sys_mknodat: invalid UTF-8 filename");
            return Err(Err::NoSys);
        }
    };

    kprint!(
        "sys_mknodat: dfd={}, filename='{}', mode=0x{:x}, dev=0x{:x}",
        _dfd,
        filename_str,
        _mode,
        _dev
    );

    let (dir_path, file_name) = split_path(filename_str);
    resolve_path(_dfd, &dir_path, 0xFFFF_FFFE)?;

    // Extract major and minor device numbers from _dev
    // In Linux, the device number is encoded as (major << 8) | minor for 8-bit minor
    // or (major << 20) | minor for 12-bit minor, but we'll use the simpler 8-bit version
    let major = (_dev >> 8) & 0xFF;
    let minor = _dev & 0xFF;

    let tmknod = TmknodMessage::new(
        0,
        0xFFFF_FFFE,
        file_name.to_string(),
        _mode,
        major,
        minor,
        0,
    );
    match tmknod.send_tmknod() {
        Ok(bytes_written) => {
            kprint!("sys_mknodat: sent {} bytes for Tmknod", bytes_written);
        }
        Err(e) => {
            kprint!("sys_mknodat: error sending Tmknod: {:?}", e);
            return Err(Err::NoSys);
        }
    }
    match RmknodMessage::read_response() {
        P9Response::Success(rmknod) => {
            kprint!("sys_mknodat: rmknod = {:?}", rmknod);
            clunk(0xFFFF_FFFE, false);
            Ok(0)
        }
        P9Response::Error(rlerror) => {
            kprint!(
                "sys_mknodat: received Rlerror for Rmknod: tag={}, ecode={}",
                rlerror.tag,
                rlerror.ecode
            );
            Ok(-(rlerror.ecode as i32) as u32)
        }
    }
}

pub fn sys_openat2(_dfd: u32, _filename: u32, _how: u32) -> Result<u32, Err> {
    // Extract and print the filename
    let filename = unsafe { core::slice::from_raw_parts(_filename as *const u8, 256) };
    let null_pos = filename
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(filename.len());
    let filename = &filename[..null_pos];

    // Convert the filename to a UTF-8 string
    let filename_str = match str::from_utf8(filename) {
        Ok(s) => s,
        Err(_) => {
            kprint!("sys_openat2: invalid UTF-8 filename");
            return Err(Err::NoSys);
        }
    };

    kprint!(
        "sys_openat2: dfd={}, filename='{}', how=0x{:x}",
        _dfd,
        filename_str,
        _how
    );

    // TODO: Implement actual file opening with 9P operations
    // For now, just return not implemented
    let msg = b"sys_openat2 not fully implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_preadv(_fd: u32, _vec: u32, _vlen: u32, _pos_low: u32) -> Result<u32, Err> {
    let msg = b"sys_preadv not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_preadv2(
    _fd: u32,
    _vec: u32,
    _vlen: u32,
    _pos_low: u32,
    _pos_high: u32,
) -> Result<u32, Err> {
    let msg = b"sys_preadv2 not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_pwrite64(_fd: u32, _buf: u32, _count: u32, _pos: u32) -> Result<u32, Err> {
    let msg = b"sys_pwrite64 not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_pwritev(_fd: u32, _vec: u32, _vlen: u32, _pos_low: u32) -> Result<u32, Err> {
    let msg = b"sys_pwritev not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_pwritev2(
    _fd: u32,
    _vec: u32,
    _vlen: u32,
    _pos_low: u32,
    _pos_high: u32,
) -> Result<u32, Err> {
    let msg = b"sys_pwritev2 not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_readv(_fd: u32, _vec: u32, _vlen: u32) -> Result<u32, Err> {
    let msg = b"sys_readv not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_ioctl(fd: u32, _cmd: u32, arg: u32) -> Result<u32, Err> {
    // For now, sys_ioctl is not implemented. Log a message for debugging.
    let msg = b"sys_ioctl not implemented";
    host_log(msg.as_ptr(), msg.len());
    let _fd = fd as i32;
    let _arg = arg as *const u8;
    // let msg = str_format!(str256, "sys_ioctl({fd}, {cmd}, 0x{arg:08x})");
    // print(&msg);
    Ok(0)
}

pub fn sys_pread64(_fd: u32, _buf: u32, _count: u32, _pos: u32) -> Result<u32, Err> {
    if _fd >= 256 {
        return Err(Err::Inval);
    }

    kprint!(
        "sys_pread64: fd={}, buf=0x{:08x}, count={}, pos={}",
        _fd,
        _buf,
        _count,
        _pos
    );

    // pread64 reads from a specific offset without changing the file position
    // We need to: save current position, seek to offset, read, restore position

    let fd_entry = get_fd(_fd);
    if fd_entry.file_desc_id == 0xFF {
        return Err(Err::BadFd);
    }

    let file_desc = get_file_desc(fd_entry.file_desc_id);
    let current_pos = file_desc.cursor;

    // Seek to requested offset (note: _pos is already the offset, not pgoffset)
    let mut updated_desc = file_desc;
    updated_desc.cursor = _pos as u64;
    update_file_desc(fd_entry.file_desc_id, updated_desc);

    // Do the read
    let result = sys_read(_fd, _buf, _count);

    // Restore original file position
    let mut restored_desc = get_file_desc(fd_entry.file_desc_id);
    restored_desc.cursor = current_pos;
    update_file_desc(fd_entry.file_desc_id, restored_desc);

    result
}

pub fn sys_readahead(_fd: u32, _offset: u32, _count: u32) -> Result<u32, Err> {
    let msg = b"sys_readahead not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_readlinkat(_dfd: u32, _pathname: u32, _buf: u32) -> Result<u32, Err> {
    let msg = b"sys_readlinkat not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_removexattr(_pathname: u32, _name: u32) -> Result<u32, Err> {
    let msg = b"sys_removexattr not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_removexattrat(_dfd: u32, _filename: u32, _name: u32) -> Result<u32, Err> {
    let msg = b"sys_removexattrat not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_lgetxattr(_pathname: u32, _name: u32, _value: u32, _size: u32) -> Result<u32, Err> {
    let msg = b"sys_lgetxattr not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_cachestat(_fd: u32, _cstat: u32, _cstat_size: u32, _flags: u32) -> Result<u32, Err> {
    let msg = b"sys_cachestat not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_renameat2(
    _olddirfd: u32,
    _oldpath: u32,
    _newdirfd: u32,
    _newpath: u32,
    _flags: u32,
) -> Result<u32, Err> {
    use crate::p9::{P9Response, RrenameatMessage, TrenameatMessage};

    if !get_p9_enabled() {
        let msg = b"sys_renameat2: p9 is not enabled";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }

    // Parse old path
    let oldpath = unsafe { core::slice::from_raw_parts(_oldpath as *const u8, 256) };
    let oldpath_len = oldpath
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(oldpath.len());
    let oldpath_str = str::from_utf8(&oldpath[..oldpath_len]).map_err(|_| Err::Inval)?;

    // Parse new path
    let newpath = unsafe { core::slice::from_raw_parts(_newpath as *const u8, 256) };
    let newpath_len = newpath
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(newpath.len());
    let newpath_str = str::from_utf8(&newpath[..newpath_len]).map_err(|_| Err::Inval)?;

    kprint!(
        "sys_renameat2: '{}' -> '{}', flags=0x{:x}",
        oldpath_str,
        newpath_str,
        _flags
    );

    // Split paths into directory and filename components
    // Use temp fids that don't conflict with CWD (0xFFFF_FFFD)
    let old_starting_fid = get_starting_fid(_olddirfd, oldpath_str)?;
    let (old_dir_path, old_name) = split_path(oldpath_str);
    let old_dir_path = normalize_path(&old_dir_path);
    do_walk(old_starting_fid, 0xFFFF_FFFE, old_dir_path)?;

    let new_starting_fid = get_starting_fid(_newdirfd, newpath_str)?;
    let (new_dir_path, new_name) = split_path(newpath_str);
    let new_dir_path = normalize_path(&new_dir_path);
    do_walk(new_starting_fid, 0xFFFF_FFFB, new_dir_path)?;

    // Send Trenameat message
    let trenameat =
        TrenameatMessage::new(0, 0xFFFF_FFFE, old_name.to_string(), 0xFFFF_FFFB, new_name);

    match trenameat.send_trenameat() {
        Ok(_) => {
            kprint!("sys_renameat2: sent Trenameat");
        }
        Err(e) => {
            kprint!("sys_renameat2: error sending Trenameat: {:?}", e);
            clunk(0xFFFF_FFFE, false);
            clunk(0xFFFF_FFFB, false);
            return Err(Err::NoSys);
        }
    }

    // Read response
    match RrenameatMessage::read_response() {
        P9Response::Success(_rrenameat) => {
            kprint!("sys_renameat2: rename successful");
            clunk(0xFFFF_FFFE, false);
            clunk(0xFFFF_FFFB, false);
            Ok(0)
        }
        P9Response::Error(rlerror) => {
            kprint!("sys_renameat2: received Rlerror: ecode={}", rlerror.ecode);
            clunk(0xFFFF_FFFE, false);
            clunk(0xFFFF_FFFB, false);
            Ok(-(rlerror.ecode as i32) as u32)
        }
    }
}

pub fn sys_sendfile64(_out_fd: u32, _in_fd: u32, _offset: u32, _count: u32) -> Result<u32, Err> {
    let msg = b"sys_sendfile64 not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_setxattr(
    _pathname: u32,
    _name: u32,
    _value: u32,
    _size: u32,
    _flags: u32,
) -> Result<u32, Err> {
    let msg = b"sys_setxattr not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_setxattrat(
    _dfd: u32,
    _filename: u32,
    _name: u32,
    _value: u32,
    _size: u32,
) -> Result<u32, Err> {
    let msg = b"sys_setxattrat not implemented";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

const F_DUPFD_CLOEXEC: u32 = 1030;

pub fn sys_fcntl64(_fd: u32, _cmd: u32, _arg: u32) -> Result<u32, Err> {
    kprint!("sys_fcntl64: _fd={} _cmd={} _arg={}", _fd, _cmd, _arg);

    // Check fd validity first (for EBADF vs EINVAL priority)
    // Check for out-of-range file descriptors
    let fd_i32 = _fd as i32;
    if fd_i32 < 0 || _fd >= 256 {
        // For negative or too large fd, return EBADF
        return Err(Err::BadFd);
    }

    let fd_entry = get_fd(_fd);
    if fd_entry.file_desc_id == 0xFF {
        return Err(Err::BadFd);
    }

    // Now check command validity
    const VALID_CMDS: &[u32] = &[
        F_DUPFD,
        F_GETFD,
        F_SETFD,
        F_GETFL,
        F_SETFL,
        F_GETLK,
        F_SETLK,
        F_SETLKW,
        F_DUPFD_CLOEXEC,
    ];

    if !VALID_CMDS.contains(&_cmd) {
        kprint!("sys_fcntl64: invalid command {}", _cmd);
        return Err(Err::Inval);
    }

    if _cmd == F_DUPFD {
        // Duplicate file descriptor with lowest available fd >= _arg
        kprint!("sys_fcntl64: F_DUPFD fd={} arg={}", _fd, _arg);

        // Find the lowest available fd >= _arg
        let start_fd = _arg.max(3); // Start from at least 3 (after stdin/stdout/stderr)
        let mut new_fd = None;
        for candidate in start_fd..256 {
            let candidate_entry = get_fd(candidate);
            if candidate_entry.file_desc_id == 0xFF {
                new_fd = Some(candidate);
                break;
            }
        }

        match new_fd {
            Some(fd) => {
                // Duplicate the file description
                update_fd(
                    fd,
                    FileDescriptor {
                        file_desc_id: fd_entry.file_desc_id,
                        cloexec: false, // F_DUPFD does not set FD_CLOEXEC
                    },
                );
                inc_refcount(fd_entry.file_desc_id);
                kprint!("sys_fcntl64: F_DUPFD allocated fd {}", fd);
                return Ok(fd);
            }
            None => {
                kprint!("sys_fcntl64: F_DUPFD no available fd");
                return Err(Err::MFile);
            }
        }
    } else if _cmd == F_GETFD {
        // Get file descriptor flags (only FD_CLOEXEC is relevant)
        let flags = if fd_entry.cloexec { FD_CLOEXEC } else { 0 };
        kprint!(
            "sys_fcntl64: F_GETFD returning {} (cloexec={})",
            flags,
            fd_entry.cloexec
        );
        return Ok(flags);
    } else if _cmd == F_SETFD {
        // Set file descriptor flags
        let new_cloexec = (_arg & FD_CLOEXEC) != 0;
        let mut updated_entry = fd_entry;
        updated_entry.cloexec = new_cloexec;
        update_fd(_fd, updated_entry);
        kprint!("sys_fcntl64: F_SETFD setting FD_CLOEXEC to {}", new_cloexec);
        return Ok(0);
    } else if _cmd == F_GETFL {
        // Get file status flags (O_RDONLY, O_WRONLY, O_RDWR, O_APPEND, O_NONBLOCK, etc.)
        let file_desc = get_file_desc(fd_entry.file_desc_id);
        let flags = file_desc.flags;
        kprint!("sys_fcntl64: F_GETFL returning 0x{:x}", flags);
        return Ok(flags);
    } else if _cmd == F_SETFL {
        // Set file status flags (only certain flags can be set: O_APPEND, O_NONBLOCK, O_ASYNC, O_DIRECT)
        // O_RDONLY, O_WRONLY, O_RDWR, and other open-time flags cannot be changed
        const O_APPEND: u32 = 0o2000; // 0x400
        const O_NONBLOCK: u32 = 0o4000; // 0x800 (also O_NDELAY)
        const O_ASYNC: u32 = 0o20000; // 0x2000
        const O_DIRECT: u32 = 0o40000; // 0x4000
        const O_NOATIME: u32 = 0o1000000; // 0x40000

        const SETTABLE_FLAGS: u32 = O_APPEND | O_NONBLOCK | O_ASYNC | O_DIRECT | O_NOATIME;

        let file_desc = get_file_desc(fd_entry.file_desc_id);
        // Keep the access mode (O_RDONLY/O_WRONLY/O_RDWR) from the original flags
        let access_mode = file_desc.mode & 0x3; // O_ACCMODE = 0x3
        // Update only the settable flags
        let new_flags = access_mode | (_arg & SETTABLE_FLAGS);

        let mut updated_desc = file_desc;
        updated_desc.flags = new_flags;
        update_file_desc(fd_entry.file_desc_id, updated_desc);

        kprint!(
            "sys_fcntl64: F_SETFL old_flags=0x{:x} new_flags=0x{:x}",
            file_desc.flags,
            new_flags
        );
        return Ok(0);
    } else if _cmd == F_DUPFD_CLOEXEC {
        let fd_entry = get_fd(_fd);
        if fd_entry.file_desc_id == 0xFF {
            return Err(Err::BadFd);
        }
        let _file_desc = get_file_desc(fd_entry.file_desc_id);
        // F_DUPFD_CLOEXEC should work like dup3, but we're not implementing the full logic here
        // For now, just duplicate the file description
        let new_fd = _arg;
        update_fd(
            new_fd,
            FileDescriptor {
                file_desc_id: fd_entry.file_desc_id,
                cloexec: true, // F_DUPFD_CLOEXEC sets FD_CLOEXEC
            },
        );
        inc_refcount(fd_entry.file_desc_id);
        kprint!("sys_fcntl64: F_DUPFD_CLOEXEC: _fd={} _arg={}", _fd, _arg);
        return Ok(_arg);
    } else if _cmd == F_GETLK {
        // Get file lock status
        // In zkVM, we don't have real file locking, so we always report F_UNLCK (no lock)
        kprint!("sys_fcntl64: F_GETLK fd={} flock_ptr=0x{:x}", _fd, _arg);

        // The _arg is a pointer to struct flock
        // struct flock {
        //     short l_type;    /* Type of lock: F_RDLCK, F_WRLCK, F_UNLCK */
        //     short l_whence;  /* How to interpret l_start: SEEK_SET, SEEK_CUR, SEEK_END */
        //     off_t l_start;   /* Starting offset for lock */
        //     off_t l_len;     /* Number of bytes to lock */
        //     pid_t l_pid;     /* PID of process blocking our lock (F_GETLK only) */
        // };

        // Check for NULL pointer
        if _arg == 0 {
            return Err(Err::Fault);
        }

        // Validate the pointer is in user memory
        const USER_MEMORY_START: u32 = 0x6800_0000;
        const USER_MEMORY_END: u32 = 0xa000_0000;
        if _arg < USER_MEMORY_START || _arg >= USER_MEMORY_END {
            return Err(Err::Fault);
        }

        // Write F_UNLCK (2) to l_type field (first 2 bytes)
        unsafe {
            let l_type_ptr = _arg as *mut u16;
            *l_type_ptr = F_UNLCK as u16;
        }

        kprint!("sys_fcntl64: F_GETLK set l_type to F_UNLCK");
        return Ok(0);
    } else if _cmd == F_SETLK || _cmd == F_SETLKW {
        // Set file lock (non-blocking or blocking)
        kprint!(
            "sys_fcntl64: F_SETLK/F_SETLKW fd={} flock_ptr=0x{:x}",
            _fd,
            _arg
        );

        // Check for NULL pointer
        if _arg == 0 {
            return Err(Err::Fault);
        }

        // Validate the pointer is in user memory
        const USER_MEMORY_START: u32 = 0x6800_0000;
        const USER_MEMORY_END: u32 = 0xa000_0000;
        if _arg < USER_MEMORY_START || _arg >= USER_MEMORY_END {
            return Err(Err::Fault);
        }

        // Read and validate the flock structure
        // struct flock layout (on 32-bit RISC-V):
        // - l_type: i16 at offset 0
        // - l_whence: i16 at offset 2
        // - l_start: i64 at offset 4 (or i32, depending on arch)
        // - l_len: i64 at offset 8 or 12
        // - l_pid: i32 at offset 12 or 16

        const SEEK_SET: u16 = 0;
        const SEEK_CUR: u16 = 1;
        const SEEK_END: u16 = 2;

        unsafe {
            let l_whence_ptr = (_arg + 2) as *const i16;
            let l_whence = *l_whence_ptr;

            // Validate l_whence
            if l_whence != SEEK_SET as i16
                && l_whence != SEEK_CUR as i16
                && l_whence != SEEK_END as i16
            {
                kprint!("sys_fcntl64: F_SETLK invalid l_whence={}", l_whence);
                return Err(Err::Inval);
            }
        }

        // In zkVM, we don't have real file locking, so we just succeed
        kprint!("sys_fcntl64: F_SETLK/F_SETLKW success (no-op in zkVM)");
        return Ok(0);
    }

    // Unsupported command
    kprint!("sys_fcntl64: unsupported command {}", _cmd);
    let msg = b"sys_fcntl64 not implemented for this cmd/arg";
    host_log(msg.as_ptr(), msg.len());
    Err(Err::NoSys)
}

pub fn sys_getdents64(fd: u32, dirp: u32, count: u32) -> Result<u32, Err> {
    if !get_p9_enabled() {
        let msg = b"sys_getdents64: p9 is not enabled";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }

    if fd >= 256 {
        return Err(Err::Inval);
    }

    let fd_entry = get_fd(fd);
    if fd_entry.file_desc_id == 0xFF {
        return Err(Err::BadFd);
    }

    let file_desc = get_file_desc(fd_entry.file_desc_id);
    let offset = file_desc.cursor;

    // Get the FID from the file description
    let fid = file_desc.fid;
    if fid == 0 {
        return Err(Err::Inval);
    }

    // XXX we will read one dir entry at a a time
    let treaddir = TreaddirMessage::new(0, fid, offset, count);
    match treaddir.send_treaddir() {
        Ok(_bytes_written) => {
            // Success
        }
        Err(_e) => {
            return Err(Err::NoSys);
        }
    }

    // Read the Rreaddir response
    match RreaddirMessage::read_response() {
        P9Response::Success(rreaddir) => {
            if rreaddir.count == 0 {
                // No more entries
                return Ok(0);
            }

            // Parse the directory entries and convert to Linux dirent64 format
            let mut offset = 0;
            let mut total_bytes = 0;
            let mut user_offset = 0;

            while offset < rreaddir.data.len() {
                // Parse 9P directory entry: qid[13] offset[8] type[1] name[s]
                if offset + 13 + 8 + 1 > rreaddir.data.len() {
                    break;
                }

                // Read QID (13 bytes)
                let qid_path = u64::from_le_bytes([
                    rreaddir.data[offset],
                    rreaddir.data[offset + 1],
                    rreaddir.data[offset + 2],
                    rreaddir.data[offset + 3],
                    rreaddir.data[offset + 4],
                    rreaddir.data[offset + 5],
                    rreaddir.data[offset + 6],
                    rreaddir.data[offset + 7],
                ]);
                let _qid_version = u32::from_le_bytes([
                    rreaddir.data[offset + 8],
                    rreaddir.data[offset + 9],
                    rreaddir.data[offset + 10],
                    rreaddir.data[offset + 11],
                ]);
                let _qid_type = rreaddir.data[offset + 12];
                offset += 13;

                // Read offset (8 bytes)
                let entry_offset = u64::from_le_bytes([
                    rreaddir.data[offset],
                    rreaddir.data[offset + 1],
                    rreaddir.data[offset + 2],
                    rreaddir.data[offset + 3],
                    rreaddir.data[offset + 4],
                    rreaddir.data[offset + 5],
                    rreaddir.data[offset + 6],
                    rreaddir.data[offset + 7],
                ]);
                offset += 8;

                // Read type (1 byte)
                let entry_type = rreaddir.data[offset];
                offset += 1;

                // Read name length (2 bytes)
                if offset + 2 > rreaddir.data.len() {
                    break;
                }
                let name_len =
                    u16::from_le_bytes([rreaddir.data[offset], rreaddir.data[offset + 1]]) as usize;
                offset += 2;

                // Read name
                if offset + name_len > rreaddir.data.len() {
                    break;
                }
                let name_bytes = &rreaddir.data[offset..offset + name_len];
                offset += name_len;

                // Convert 9P type to Linux DT_* type
                let d_type = match entry_type {
                    0 => DT_DIR, // Directory
                    1 => DT_REG, // Regular file
                    2 => DT_REG, // Append-only file
                    3 => DT_REG, // Exclusive-use file
                    4 => DT_REG, // Mounted file
                    5 => DT_REG, // Authentication file
                    6 => DT_REG, // Temporary file
                    7 => DT_LNK, // Symbolic link
                    _ => DT_UNKNOWN,
                };

                // Calculate record length (align to 8-byte boundary)
                // C compiler adds padding to align d_name, so structure is 24 bytes
                let c_struct_size = 24; // 8+8+2+1+5(padding) = 24 bytes in C
                let name_len_aligned = (name_len + 7) & !7;
                let reclen = (c_struct_size + name_len_aligned) as u16;

                // Check if we have enough space in user buffer
                if user_offset + reclen as usize > count as usize {
                    break;
                }

                // Write Linux dirent64 structure directly to user buffer
                unsafe {
                    let user_ptr = (dirp + user_offset as u32) as *mut u8;

                    // Write structure fields directly to buffer at correct offsets
                    // d_ino at offset 0-7
                    let d_ino_ptr = user_ptr as *mut u64;
                    *d_ino_ptr = qid_path;

                    // d_off at offset 8-15
                    let next_dirent_offset = (user_offset + reclen as usize + 7) & !7;
                    let d_off_ptr = user_ptr.add(8) as *mut i64;
                    *d_off_ptr = next_dirent_offset as i64;

                    // d_reclen at offset 16-17
                    let d_reclen_ptr = user_ptr.add(16) as *mut u16;
                    *d_reclen_ptr = reclen;

                    // d_type at offset 18
                    let d_type_ptr = user_ptr.add(18);
                    *d_type_ptr = d_type;

                    // Copy name (null-terminated) after the structure
                    // C compiler adds padding between d_type and d_name for alignment
                    let name_ptr = user_ptr.add(19);

                    for (i, &byte) in name_bytes.iter().enumerate().take(name_len) {
                        *name_ptr.add(i) = byte;
                    }
                    *name_ptr.add(name_len) = 0; // null terminator

                    // Zero out padding
                    for i in (name_len + 1)..name_len_aligned {
                        *name_ptr.add(i) = 0;
                    }
                }

                // Update the shared cursor in the file description
                let mut updated_desc = get_file_desc(fd_entry.file_desc_id);
                updated_desc.cursor = entry_offset;
                update_file_desc(fd_entry.file_desc_id, updated_desc);
                // Ensure next dirent is 8-byte aligned
                user_offset = (user_offset + reclen as usize + 7) & !7;
                total_bytes += reclen as usize;
            }

            Ok(total_bytes as u32)
        }
        P9Response::Error(_rlerror) => Err(Err::NoSys),
    }
}

#[allow(dead_code)]
// instead of wnames we take a path
fn do_walk(
    starting_fid: u32,
    target_fid: u32,
    wnames: Vec<String>,
) -> Result<(u32, Vec<Qid>), Err> {
    kprint!(
        "do_walk: starting_fid={}, target_fid={}, wnames={:?}",
        starting_fid,
        target_fid,
        wnames
    );

    let wnames_len = wnames.len();
    let wnames_clone = wnames.clone();
    let twalk = crate::p9::TwalkMessage::new(0, starting_fid, target_fid, wnames);
    match twalk.send_twalk() {
        Ok(bytes_written) => {
            kprint!("do_walk: bytes_written = {}", bytes_written);
        }
        Err(e) => {
            kprint!("do_walk: error = {:?}", e);
            return Err(Err::NoSys);
        }
    }
    match RwalkMessage::read_response() {
        P9Response::Success(rwalk) => {
            if rwalk.wqids.len() != wnames_len {
                clunk(target_fid, false);
                Err(Err::FileNotFound)
            } else {
                // Check if the final component is a symlink
                if let Some(last_qid) = rwalk.wqids.last()
                    && last_qid.is_symlink()
                {
                    kprint!("do_walk: final component is a symlink, resolving...");

                    // Read the symlink contents
                    let symlink_target = read_symlink(target_fid)?;
                    kprint!("do_walk: symlink target = '{}'", symlink_target);
                    let wnames = wnames_clone;
                    // Resolve the target path relative to the symlink's directory
                    let resolved_path = resolve_symlink_target(&symlink_target, &wnames)?;
                    kprint!("do_walk: resolved path = '{}'", resolved_path);

                    // Close the current target_fid since we're going to walk to a new path
                    clunk(target_fid, false);

                    // Walk to the resolved path
                    let resolved_wnames = normalize_path(&resolved_path);
                    return do_walk(starting_fid, target_fid, resolved_wnames);
                }
                Ok((0, rwalk.wqids))
            }
        }
        P9Response::Error(rlerror) => {
            kprint!(
                "do_walk: received Rlerror for walk operation: tag={}, ecode={}",
                rlerror.tag,
                rlerror.ecode
            );
            Err(Err::NoSys)
        }
    }
}

/// Read the contents of a symlink
fn read_symlink(fid: u32) -> Result<String, Err> {
    // Use Treadlink to read the symlink contents directly
    let treadlink = TreadlinkMessage::new(0, fid);
    match treadlink.send_treadlink() {
        Ok(_bytes_written) => {
            kprint!("read_symlink: sent treadlink request");
        }
        Err(e) => {
            kprint!("read_symlink: error sending treadlink: {:?}", e);
            return Err(Err::NoSys);
        }
    }

    match RreadlinkMessage::read_response() {
        P9Response::Success(rreadlink) => {
            kprint!("read_symlink: readlink target = '{}'", rreadlink.target);
            Ok(rreadlink.target)
        }
        P9Response::Error(rlerror) => {
            kprint!(
                "read_symlink: error reading symlink: tag={}, ecode={}",
                rlerror.tag,
                rlerror.ecode
            );
            Err(Err::NoSys)
        }
    }
}

/// Resolve a symlink target path relative to the symlink's directory
fn resolve_symlink_target(target: &str, original_wnames: &[String]) -> Result<String, Err> {
    if target.starts_with('/') {
        // Absolute path - use as-is
        Ok(target.to_string())
    } else {
        // Relative path - resolve relative to the symlink's directory
        // Build the directory path from the original wnames (excluding the last component)
        let mut dir_path = String::new();
        if original_wnames.len() > 1 {
            // There are parent directories
            for (i, component) in original_wnames.iter().enumerate() {
                if i == original_wnames.len() - 1 {
                    // Skip the last component (the symlink itself)
                    break;
                }
                dir_path.push('/');
                dir_path.push_str(component);
            }
        } else {
            // Symlink is in the root directory
            dir_path.push('/');
        }

        // Combine directory path with target
        let resolved = if dir_path.ends_with('/') {
            format!("{}{}", dir_path, target)
        } else {
            format!("{}/{}", dir_path, target)
        };

        kprint!(
            "resolve_symlink_target: dir_path='{}', target='{}', resolved='{}'",
            dir_path,
            target,
            resolved
        );
        Ok(resolved)
    }
}

/// Split a path into directory and filename components
/// also supports relative paths
fn split_path(path: &str) -> (String, String) {
    // Remove any trailing slashes except for root "/"
    let trimmed = if path != "/" {
        path.trim_end_matches('/').to_string()
    } else {
        path.to_string()
    };

    // Find the last slash in the trimmed path
    if let Some(pos) = trimmed.rfind('/') {
        let dir = if pos == 0 {
            // Path is like "/foo" or "/"
            "/".to_string()
        } else {
            // Path is like "/foo/bar" or "foo/bar"
            trimmed[..pos].to_string()
        };
        let filename = trimmed[pos + 1..].to_string();
        (dir, filename)
    } else {
        // No slash found, so it's a relative filename in current directory
        (".".to_string(), trimmed)
    }
}

pub fn sys_openat(_dfd: u32, _filename: u32, _flags: u32, _mode: u32) -> Result<u32, Err> {
    // Extract and print the filename
    let filename = unsafe { core::slice::from_raw_parts(_filename as *const u8, 256) };
    let null_pos = filename
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(filename.len());
    let filename = &filename[..null_pos];

    let filename_str = match str::from_utf8(filename) {
        Ok(s) => s,
        Err(_) => {
            kprint!("sys_openat: invalid UTF-8 filename");
            return Err(Err::NoSys);
        }
    };
    do_openat(_dfd, filename_str, _flags, _mode)
}

const AT_FDCWD: i32 = -100; // -100i32 as u32

fn get_starting_fid(_dfd: u32, filename_str: &str) -> Result<u32, Err> {
    // If the pathname given in pathname is relative, then it is interpreted relative to the directory referred to by the file descriptor dirfd (rather than relative to the current working directory of the calling process, as is done by open(2) for a relative pathname).
    // If pathname is relative and dirfd is the special value AT_FDCWD, then pathname is interpreted relative to the current working directory of the calling process (like open(2)).
    let _dfd_i32 = _dfd as i32;
    if filename_str.starts_with("/") {
        // Absolute paths ignore dirfd - always use root
        Ok(get_root_fid())
    } else if _dfd_i32 == AT_FDCWD {
        // AT_FDCWD case
        Ok(get_cwd_fid())
    } else {
        // For relative paths, dirfd must be valid
        // Check if dfd is in range (including negative values cast to u32)
        if _dfd_i32 < 0 || _dfd >= 256 {
            kprint!(
                "get_starting_fid: invalid dfd={} (as i32={})",
                _dfd,
                _dfd_i32
            );
            return Err(Err::BadFd);
        }

        let fd_entry = get_fd(_dfd);
        if fd_entry.file_desc_id == 0xFF {
            kprint!("get_starting_fid: dfd={} is not open", _dfd);
            return Err(Err::BadFd);
        }

        // Check if dfd is a directory (required for relative paths)
        let file_desc = get_file_desc(fd_entry.file_desc_id);
        if !file_desc.is_dir {
            kprint!(
                "get_starting_fid: dfd={} is not a directory (is_dir=false)",
                _dfd
            );
            return Err(Err::NotDir);
        }

        Ok(file_desc.fid)
    }
}

#[allow(dead_code)]
fn get_dir_fid_into_temp_fid(dfd: u32, filename_str: &str) -> Result<(u32, String), Err> {
    let starting_fid = get_starting_fid(dfd, filename_str)?;
    let (dir_path, file_name) = split_path(filename_str);
    let dir_path = normalize_path(&dir_path);
    // Use temp fid that doesn't conflict with CWD (0xFFFF_FFFD)
    do_walk(starting_fid, 0xFFFF_FFFB, dir_path)?;
    Ok((0xFFFF_FFFB, file_name))
}

fn resolve_file_to_fid(dir_fid: u32, new_fid: u32, filename: &str) -> Result<u32, Err> {
    do_walk(dir_fid, new_fid, vec![filename.to_string()])?;
    Ok(0)
}

fn resolve_path(dfd: u32, filename_str: &str, into_fid: u32) -> Result<u32, Err> {
    let starting_fid = get_starting_fid(dfd, filename_str)?;

    let (dir_path, file_name) = split_path(filename_str);
    let dir_path = normalize_path(&dir_path);
    let mut file_path = dir_path.clone();
    file_path.push(file_name.clone());

    if do_walk(starting_fid, into_fid, file_path).is_ok() {
        Ok(0)
    } else {
        Err(Err::FileNotFound)
    }
}

fn do_openat(dfd: u32, filename_str: &str, _flags: u32, mode: u32) -> Result<u32, Err> {
    // Convert the filename to a UTF-8 s
    kprint!(
        "sys_openat: filename='{}' dfd={} flags=0x{:x} mode=0x{:x}",
        filename_str,
        dfd,
        _flags,
        mode
    );

    const O_CREAT: u32 = 0o100;
    const O_EXCL: u32 = 0o200;
    const O_TRUNC: u32 = 0o1000;
    let p9_flags = if (_flags & 0o3) == 0o0 {
        0
    }
    // O_RDONLY
    else if (_flags & 0o3) == 0o1 {
        1
    }
    // O_WRONLY
    else {
        2
    }; // O_RDWR

    let file_fid = find_free_fd()?;

    if resolve_path(dfd, filename_str, file_fid).is_ok() {
        if (_flags & O_CREAT) == O_CREAT && (_flags & O_EXCL) == O_EXCL {
            kprint!("sys_openat: O_CREAT and O_EXCL flags detected, file already exists");
            return Err(Err::FileExists);
        }
        kprint!("sys_openat: file exists, flags=0x{:x}", _flags);

        let tlopen = TlopenMessage::new(0, file_fid, p9_flags);

        match tlopen.send_tlopen() {
            Ok(bytes_written) => {
                kprint!("sys_openat: sent {} bytes for Tlopen", bytes_written);

                // Read the response
                match RlopenMessage::read_response() {
                    P9Response::Success(rlopen) => {
                        kprint!("sys_openat: received Rlopen: {:?}", rlopen);
                        let is_directory = rlopen.qid.is_dir();
                        set_fd(file_fid, file_fid);
                        // Update the file description with the correct mode, is_dir, and flags
                        let fd_entry = get_fd(file_fid);
                        let mut file_desc = get_file_desc(fd_entry.file_desc_id);
                        file_desc.mode = p9_flags;
                        file_desc.is_dir = is_directory;
                        file_desc.flags = _flags;
                        update_file_desc(fd_entry.file_desc_id, file_desc);

                        // Handle O_TRUNC: truncate file to 0 bytes after opening
                        if (_flags & O_TRUNC) == O_TRUNC {
                            kprint!("sys_openat: O_TRUNC detected, truncating file to 0 bytes");
                            let valid = P9SetattrMask::Size as u32;
                            let tsetattr = TsetattrMessage::new(
                                0,        // tag
                                file_fid, // fid
                                valid,    // valid mask (only Size)
                                0,        // mode (not changing)
                                0,        // uid (not changing)
                                0,        // gid (not changing)
                                0,        // size (truncate to 0)
                                0,        // atime_sec (not changing)
                                0,        // atime_nsec (not changing)
                                0,        // mtime_sec (not changing)
                                0,        // mtime_nsec (not changing)
                            );

                            match tsetattr.send_tsetattr() {
                                Ok(_) => match RsetattrMessage::read_response() {
                                    P9Response::Success(_) => {
                                        kprint!("sys_openat: file truncated successfully");
                                    }
                                    P9Response::Error(rlerror) => {
                                        kprint!(
                                            "sys_openat: error truncating file: ecode={}",
                                            rlerror.ecode
                                        );
                                        clunk(file_fid, false);
                                        return Ok(-(rlerror.ecode as i32) as u32);
                                    }
                                },
                                Err(e) => {
                                    kprint!("sys_openat: error sending Tsetattr: {:?}", e);
                                    clunk(file_fid, false);
                                    return Err(Err::NoSys);
                                }
                            }
                        }

                        Ok(file_fid)
                    }
                    P9Response::Error(rlerror) => {
                        clunk(file_fid, false);
                        kprint!(
                            "sys_openat: received Rlerror for Rlopen: tag={}, ecode={}",
                            rlerror.tag,
                            rlerror.ecode
                        );
                        Ok(-(rlerror.ecode as i32) as u32)
                    }
                }
            }
            Err(e) => {
                kprint!("sys_openat: error sending Tlopen: {:?}", e);
                Err(Err::NoSys)
            }
        }
    } else if _flags & O_CREAT == O_CREAT {
        kprint!("sys_openat: O_CREAT flag detected, attempting file creation");
        let (dir_path, file_name) = split_path(filename_str);
        resolve_path(dfd, &dir_path, file_fid)?;

        let p9_mode = mode & 0o777; // Basic permission bits
        // Create the file using Tlcreate
        let tlcreate =
            TlcreateMessage::new(0, file_fid, file_name.to_string(), p9_flags, p9_mode, 0); // flags=p9_flags, mode=p9_mode, gid=0
        kprint!("sys_openat: tlcreate = {:?}", tlcreate);
        match tlcreate.send_tlcreate() {
            Ok(bytes_written) => {
                kprint!("sys_openat: sent {} bytes for Tlcreate", bytes_written);

                // Read the response
                match RlcreateMessage::read_response() {
                    P9Response::Success(rlcreate) => {
                        kprint!("sys_openat: received Rlcreate: {:?}", rlcreate);
                        let is_directory = rlcreate.qid.is_dir();
                        set_fd(file_fid, file_fid);
                        // Update the file description with the correct mode, is_dir, and flags
                        let fd_entry = get_fd(file_fid);
                        let mut file_desc = get_file_desc(fd_entry.file_desc_id);
                        file_desc.mode = p9_flags;
                        file_desc.is_dir = is_directory;
                        file_desc.flags = _flags;
                        update_file_desc(fd_entry.file_desc_id, file_desc);
                        Ok(file_fid)
                    }
                    P9Response::Error(rlerror) => {
                        clunk(file_fid, false);
                        kprint!(
                            "sys_openat: received Rlerror for Rlcreate: tag={}, ecode={}",
                            rlerror.tag,
                            rlerror.ecode
                        );
                        Ok(-(rlerror.ecode as i32) as u32)
                    }
                }
            }
            Err(e) => {
                kprint!("sys_openat: error sending Tlcreate: {:?}", e);
                clunk(file_fid, false);
                Err(Err::NoSys)
            }
        }
    } else {
        kprint!("sys_openat: file not found");
        Err(Err::FileNotFound)
    }
}

// This implementation mimics the behavior of the POSIX dirname utility.
// It returns the parent directory of the given path, handling edge cases.
#[allow(dead_code)]
fn dirname(path: &str) -> String {
    // Remove trailing slashes except for root
    let path = path.trim_end_matches('/');

    // If the path is empty after trimming, return "/"
    if path.is_empty() {
        return "/".to_string();
    }

    // Find the last slash in the path
    if let Some(pos) = path.rfind('/') {
        if pos == 0 {
            // Path is like "/foo" or "/"
            "/".to_string()
        } else {
            // Path is like "/foo/bar" or "foo/bar"
            path[..pos].to_string()
        }
    } else {
        // No slash found, so return "/"
        "/".to_string()
    }
}

fn normalize_path(path: &str) -> Vec<String> {
    let split: Vec<String> = path.split("/").map(|s| s.to_string()).collect();
    // remove ./
    let mut new_path = Vec::new();
    for s in split {
        if s == "." {
            continue;
        }
        new_path.push(s);
    }
    // remove all empty strings in new_path
    new_path.retain(|s| !s.is_empty());
    new_path
}

#[allow(dead_code)]
fn get_file_size(starting_fid: u32, path: &str, depth: u32) -> Result<u64, Err> {
    if depth > 40 {
        // don't let it recurse too much
        kprint!("get_file_size: depth too deep, returning error");
        return Err(Err::NoSys);
    }
    let wnames = normalize_path(path);
    do_walk(starting_fid, 0xFFFF_FFFE, wnames)?;

    let tgetattr = crate::p9::TgetattrMessage::new(0, 0xFFFF_FFFE, P9_GETATTR_ALL);
    match tgetattr.send_tgetattr() {
        Ok(bytes_written) => {
            kprint!("sys_statx: bytes_written = {}", bytes_written);
        }
        Err(e) => {
            kprint!("sys_statx: error = {:?}", e);
        }
    }
    match RgetattrMessage::read_response() {
        P9Response::Success(rgetattr) => {
            kprint!("sys_statx: rgetattr = {:?}", rgetattr);
            clunk(0xFFFF_FFFE, false);
            Ok(rgetattr.size)
        }
        P9Response::Error(rlerror) => {
            kprint!(
                "sys_statx: received Rlerror for getattr operation: tag={}, ecode={}",
                rlerror.tag,
                rlerror.ecode
            );
            clunk(0xFFFF_FFFE, false);
            Err(Err::NoSys)
        }
    }
}

pub fn sys_statx(
    _dfd: u32,
    _filename: u32,
    _flags: u32,
    _mask: u32,
    _statxbuf: u32,
) -> Result<u32, Err> {
    if !get_p9_enabled() {
        let msg = b"sys_statx: p9 is not enabled";
        host_log(msg.as_ptr(), msg.len());
        return Err(Err::NoSys);
    }

    // let's print the filename we're trying to stat, it's a u32 pointer
    // we need to search for the null-terminator
    // XXX this is ugly
    kprint!("sys_statx: _statxbuf = {:08x}", _statxbuf);
    let filename = unsafe { core::slice::from_raw_parts(_filename as *const u8, 256) };
    let null_pos = filename.iter().position(|&b| b == 0).unwrap();
    let filename = &filename[..null_pos];
    // convert the filename, it's a 0-terminated utf-8 string
    let filename = str::from_utf8(filename).unwrap();
    kprint!("sys_statx: filename = {}", filename);

    // AT_EMPTY_PATH flag (0x1000) - if filename is empty, stat the fd itself
    const AT_EMPTY_PATH: u32 = 0x1000;
    const AT_SYMLINK_NOFOLLOW: u32 = 0x100;

    if (_flags & AT_EMPTY_PATH) != 0 && _dfd < 3 {
        // FIXME this doesn't return a statx buffer, it returns 0
        return Ok(0);
    }
    let target_fid = if (_flags & AT_EMPTY_PATH) != 0 && filename.is_empty() {
        // Stat the file descriptor itself
        kprint!("sys_statx: AT_EMPTY_PATH set, statting fd {}", _dfd);
        let fd_entry = get_fd(_dfd);
        if fd_entry.file_desc_id == 0xFF {
            kprint!("sys_statx: invalid fd {}", _dfd);
            return Err(Err::Inval);
        }
        let file_desc = get_file_desc(fd_entry.file_desc_id);
        file_desc.fid
    } else if (_flags & AT_SYMLINK_NOFOLLOW) != 0 {
        // Don't follow symlinks - use Twalk directly on the final component
        kprint!("sys_statx: AT_SYMLINK_NOFOLLOW set, not following symlinks");
        let starting_fid = get_starting_fid(_dfd, filename)?;
        let (dir_path, file_name) = split_path(filename);

        // For relative paths, use the dfd directly as the directory
        let dir_fid = if dir_path.is_empty() || dir_path == "." {
            starting_fid
        } else {
            let dir_path = normalize_path(&dir_path);
            do_walk(starting_fid, 0xFFFF_FFFB, dir_path)?;
            0xFFFF_FFFB
        };

        // Walk to the final component without following symlinks
        let file_path = vec![file_name];
        let twalk = crate::p9::TwalkMessage::new(0, dir_fid, 0xFFFF_FFFE, file_path);
        match twalk.send_twalk() {
            Ok(_) => match RwalkMessage::read_response() {
                P9Response::Success(rwalk) => {
                    if rwalk.wqids.len() != 1 {
                        clunk(0xFFFF_FFFE, false);
                        if dir_fid != starting_fid {
                            clunk(dir_fid, false);
                        }
                        return Err(Err::FileNotFound);
                    }
                    // Clean up directory fid if we allocated one
                    if dir_fid != starting_fid {
                        clunk(dir_fid, false);
                    }
                }
                P9Response::Error(_) => {
                    if dir_fid != starting_fid {
                        clunk(dir_fid, false);
                    }
                    return Err(Err::FileNotFound);
                }
            },
            Err(_) => {
                if dir_fid != starting_fid {
                    clunk(dir_fid, false);
                }
                return Err(Err::NoSys);
            }
        }
        0xFFFF_FFFE
    } else {
        // Normal path resolution (follows symlinks)
        // Use temp fids that don't conflict with CWD (0xFFFF_FFFD)
        let starting_fid = get_starting_fid(_dfd, filename)?;
        let (dir_path, file_name) = split_path(filename);
        let dir_path = normalize_path(&dir_path);
        do_walk(starting_fid, 0xFFFF_FFFB, dir_path)?;
        resolve_file_to_fid(0xFFFF_FFFB, 0xFFFF_FFFE, &file_name)?;
        clunk(0xFFFF_FFFB, false); // Clean up directory fid
        0xFFFF_FFFE
    };

    let tgetattr = crate::p9::TgetattrMessage::new(0, target_fid, P9_GETATTR_ALL);
    match tgetattr.send_tgetattr() {
        Ok(bytes_written) => {
            kprint!("sys_statx: bytes_written = {}", bytes_written);
        }
        Err(e) => {
            kprint!("sys_statx: error = {:?}", e);
        }
    }
    match RgetattrMessage::read_response() {
        P9Response::Success(rgetattr) => {
            kprint!("sys_statx: rgetattr = {:?}", rgetattr);

            // Convert 9P RgetattrMessage to Linux statx structure
            let statx = convert_rgetattr_to_statx(&rgetattr);

            // Write the statx structure to the user buffer using copy_to_user
            let statx_bytes = unsafe {
                core::slice::from_raw_parts(
                    &statx as *const Statx as *const u8,
                    core::mem::size_of::<Statx>(),
                )
            };
            let bytes_copied = crate::kernel::copy_to_user(
                _statxbuf as *mut u8,
                statx_bytes.as_ptr(),
                core::mem::size_of::<Statx>(),
            );
            if bytes_copied == 0 {
                kprint!("sys_statx: failed to copy statx structure to user memory");
                return Err(Err::NoSys);
            }

            kprint!("sys_statx: successfully filled statx buffer");
            if target_fid == 0xFFFF_FFFE {
                clunk(0xFFFF_FFFE, false);
            }
            Ok(0) // Success
        }
        P9Response::Error(rlerror) => {
            kprint!(
                "sys_statx: received Rlerror for getattr operation: tag={}, ecode={}",
                rlerror.tag,
                rlerror.ecode
            );
            if target_fid == 0xFFFF_FFFE {
                clunk(0xFFFF_FFFE, false);
            }
            Ok(-(rlerror.ecode as i32) as u32)
        }
    }
}

/// Convert 9P RgetattrMessage to Linux statx structure
fn convert_rgetattr_to_statx(rgetattr: &RgetattrMessage) -> Statx {
    // Determine which fields are valid based on the valid mask
    let mut stx_mask = 0u32;

    // Map 9P valid mask to statx mask
    if (rgetattr.valid & P9_GETATTR_MODE) != 0 {
        stx_mask |= STATX_TYPE | STATX_MODE;
    }
    if (rgetattr.valid & P9_GETATTR_NLINK) != 0 {
        stx_mask |= STATX_NLINK;
    }
    if (rgetattr.valid & P9_GETATTR_UID) != 0 {
        stx_mask |= STATX_UID;
    }
    if (rgetattr.valid & P9_GETATTR_GID) != 0 {
        stx_mask |= STATX_GID;
    }
    if (rgetattr.valid & P9_GETATTR_ATIME) != 0 {
        stx_mask |= STATX_ATIME;
    }
    if (rgetattr.valid & P9_GETATTR_MTIME) != 0 {
        stx_mask |= STATX_MTIME;
    }
    if (rgetattr.valid & P9_GETATTR_CTIME) != 0 {
        stx_mask |= STATX_CTIME;
    }
    if (rgetattr.valid & P9_GETATTR_INO) != 0 {
        stx_mask |= STATX_INO;
    }
    if (rgetattr.valid & P9_GETATTR_SIZE) != 0 {
        stx_mask |= STATX_SIZE;
    }
    if (rgetattr.valid & P9_GETATTR_BLOCKS) != 0 {
        stx_mask |= STATX_BLOCKS;
    }
    if (rgetattr.valid & P9_GETATTR_BTIME) != 0 {
        stx_mask |= STATX_BTIME;
    }

    // Use the mode field directly from 9P (only if mode is valid)
    let stx_mode = if (rgetattr.valid & P9_GETATTR_MODE) != 0 {
        rgetattr.mode as u16 // The mode field already contains file type + permissions
    } else {
        0 // Mode not valid, set to 0
    };

    Statx {
        stx_mask,
        stx_blksize: rgetattr.blksize as u32,
        stx_attributes: 0, // No special attributes for now
        stx_nlink: rgetattr.nlink as u32,
        stx_uid: rgetattr.uid,
        stx_gid: rgetattr.gid,
        stx_mode,
        __spare0: [0],
        stx_ino: rgetattr.qid.path, // Use QID path as inode number
        stx_size: rgetattr.size,
        stx_blocks: rgetattr.blocks,
        stx_attributes_mask: 0, // No special attributes supported
        stx_atime: StatxTimestamp {
            tv_sec: rgetattr.atime_sec as i64,
            tv_nsec: rgetattr.atime_nsec as u32,
            __reserved: 0,
        },
        stx_btime: StatxTimestamp {
            tv_sec: rgetattr.btime_sec as i64,
            tv_nsec: rgetattr.btime_nsec as u32,
            __reserved: 0,
        },
        stx_ctime: StatxTimestamp {
            tv_sec: rgetattr.ctime_sec as i64,
            tv_nsec: rgetattr.ctime_nsec as u32,
            __reserved: 0,
        },
        stx_mtime: StatxTimestamp {
            tv_sec: rgetattr.mtime_sec as i64,
            tv_nsec: rgetattr.mtime_nsec as u32,
            __reserved: 0,
        },
        stx_rdev_major: (rgetattr.rdev >> 8) as u32,
        stx_rdev_minor: (rgetattr.rdev & 0xFF) as u32,
        stx_dev_major: 0, // Not available from 9P
        stx_dev_minor: 0, // Not available from 9P
        stx_mnt_id: 0,    // Not available from 9P
        stx_dio_mem_align: 0,
        stx_dio_offset_align: 0,
        __spare3: [0; 12],
    }
}

// make a fd
pub fn find_free_fd() -> Result<u32, Err> {
    let mut fd = 0;
    while fd < 256 && get_fd(fd).file_desc_id != 0xFF {
        fd += 1;
    }
    if fd >= 256 {
        kprint!("find_free_fd: no free file descriptors available");
        return Err(Err::MFile);
    }
    Ok(fd)
}

#[allow(dead_code)]
pub fn set_fd(fd: u32, fid: u32) {
    // Allocate a new file description
    let desc_id = find_free_file_desc().unwrap_or(0);
    update_file_desc(
        desc_id,
        FileDescription {
            fid,
            cursor: 0,
            is_dir: false,
            mode: 0xFFFF_FFFF,
            flags: 0,
            refcount: 1, // Initial reference
        },
    );

    // Point the fd to this file description
    update_fd(
        fd,
        FileDescriptor {
            file_desc_id: desc_id,
            cloexec: false, // New file descriptors start with FD_CLOEXEC unset
        },
    );
}

#[repr(C)]
pub struct IoVec {
    pub iov_base: *mut u8,
    pub iov_len: usize,
}

pub fn clunk(fid: u32, is_cwd_clunking_allowed: bool) {
    // Check if we're trying to clunk the CWD FID without permission
    if fid == get_cwd_fid() && !is_cwd_clunking_allowed {
        kpanic!("Attempted to clunk CWD FID {} without permission", fid);
    }

    let tclunk = crate::p9::TclunkMessage::new(0, fid);
    kprint!("clunk: sending TclunkMessage for fid {}", fid);
    tclunk.send_tclunk().unwrap();
    let rclunk = RclunkMessage::read_response();
    match rclunk {
        P9Response::Success(rclunk) => {
            kprint!("clunk {}: = {:?}", fid, rclunk);
        }
        P9Response::Error(rlerror) => {
            kprint!(
                "clunk: received Rlerror for clunk operation: tag={}, ecode={}",
                rlerror.tag,
                rlerror.ecode
            );
        }
    }
}

pub fn attach_to_p9() {
    let msg = TversionMessage::default_9p2000l(1);

    // Send the message via host_write to file descriptor 1
    match msg.send_tversion() {
        Ok(bytes_written) => {
            kprint!("Sent {} bytes", bytes_written);
        }
        Err(_e) => {
            kprint!("Failed to send Tversion");
        }
    }
    match RversionMessage::read_response() {
        P9Response::Success(rversion) => {
            kprint!("Successfully read Rversion: {:?}", rversion);
            // Process the message
        }
        P9Response::Error(rlerror) => {
            kpanic!(
                "Received Rlerror for Rversion: tag={}, ecode={}",
                rlerror.tag,
                rlerror.ecode
            );
        }
    }
    // now we attach with Tattach, P9_NOFID as afid, to /risc0-root
    let tattach = TattachMessage::new(
        0,
        0,
        crate::p9::constants::P9_NOFID,
        "root".to_string(),
        "/risc0-root".to_string(),
    );
    match tattach.send_tattach() {
        Ok(bytes_written) => {
            kprint!("Sent {} bytes", bytes_written);
        }
        Err(e) => {
            kpanic!("Failed to send Tattach: {:?}", e);
        }
    }
    // and wait for Rattach
    match RattachMessage::read_response() {
        P9Response::Success(rattach) => {
            kprint!("Received Rattach: {:?}", rattach);
            set_cwd_str("/".to_string());
            set_cwd_fid(0);
            set_root_fid(0);
        }
        P9Response::Error(rlerror) => {
            kpanic!(
                "Received Rlerror for Rattach: tag={}, ecode={}",
                rlerror.tag,
                rlerror.ecode
            );
        }
    }
}
