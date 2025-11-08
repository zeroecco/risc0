// Copyright 2025 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{
    cell::RefCell,
    fmt::Debug,
    marker::PhantomData,
    rc::Rc,
    sync::{
        atomic::{AtomicU64, Ordering},
        OnceLock,
    },
};

use anyhow::{bail, Context as _, Result};
use cust::{
    device::DeviceAttribute,
    memory::{DeviceCopy, DevicePointer, GpuBuffer},
    prelude::*,
    stream::{Stream, StreamFlags},
};
use parking_lot::{ReentrantMutex, ReentrantMutexGuard};
use risc0_core::{
    field::{
        baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem},
        Elem, ExtElem, RootsOfUnity,
    },
    scope,
};
use risc0_sys::{cuda::*, ffi_wrap};

// CUDA memory advice hints for optimizing memory access patterns
#[repr(u32)]
#[allow(dead_code)]
enum CudaMemAdvise {
    SetReadMostly = 1,
    UnsetReadMostly = 2,
    SetPreferredLocation = 3,
    UnsetPreferredLocation = 4,
    SetAccessedBy = 5,
    UnsetAccessedBy = 6,
}

extern "C" {
    fn cudaMemAdvise(
        devPtr: *const std::os::raw::c_void,
        count: usize,
        advice: u32,
        device: std::os::raw::c_int,
    ) -> std::os::raw::c_int;

    #[allow(dead_code)]
    fn cudaMemcpyAsync(
        dst: *mut std::os::raw::c_void,
        src: *const std::os::raw::c_void,
        count: usize,
        kind: u32,
        stream: *mut std::os::raw::c_void,
    ) -> std::os::raw::c_int;

    #[allow(dead_code)]
    fn cudaStreamSynchronize(stream: *mut std::os::raw::c_void) -> std::os::raw::c_int;
}

#[repr(u32)]
#[allow(dead_code)]
enum CudaMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
}

use super::{tracker, Buffer, Hal};
use crate::{
    core::{
        digest::Digest,
        hash::{
            poseidon2::Poseidon2HashSuite, poseidon_254::Poseidon254HashSuite,
            sha::Sha256HashSuite, HashSuite,
        },
        log2_ceil,
    },
    FRI_FOLD,
};

// The GPU becomes unstable as the number of concurrent provers grow.
pub fn singleton() -> &'static ReentrantMutex<()> {
    static ONCE: OnceLock<ReentrantMutex<()>> = OnceLock::new();
    ONCE.get_or_init(|| ReentrantMutex::new(()))
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct DeviceElem(pub BabyBearElem);

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct DeviceExtElem(pub BabyBearExtElem);

unsafe impl DeviceCopy for DeviceExtElem {}

pub trait CudaHash {
    /// Create a hash implementation
    fn new() -> Self
    where
        Self: Sized;

    /// Run the hash_fold function
    fn hash_fold(&self, io: &BufferImpl<Digest>, output_size: usize);

    /// Run the hash_rows function
    fn hash_rows(&self, output: &BufferImpl<Digest>, matrix: &BufferImpl<BabyBearElem>);

    /// Return the HashSuite
    fn get_hash_suite(&self) -> &HashSuite<BabyBear>;
}

pub struct CudaHashSha256 {
    suite: HashSuite<BabyBear>,
}

impl CudaHash for CudaHashSha256 {
    fn new() -> Self {
        CudaHashSha256 {
            suite: Sha256HashSuite::new_suite(),
        }
    }

    fn hash_fold(&self, io: &BufferImpl<Digest>, output_size: usize) {
        let input = io.as_device_ptr_with_offset(2 * output_size);
        let output = io.as_device_ptr_with_offset(output_size);

        extern "C" {
            fn risc0_zkp_cuda_sha_fold(
                output: DevicePointer<u8>,
                input: DevicePointer<u8>,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        execute_kernel(|| {
            ffi_wrap(|| unsafe { risc0_zkp_cuda_sha_fold(output, input, output_size as u32) })
        })
        .unwrap();
    }

    fn hash_rows(&self, output: &BufferImpl<Digest>, matrix: &BufferImpl<BabyBearElem>) {
        let row_size = output.size();
        let col_size = matrix.size() / output.size();
        assert_eq!(matrix.size(), col_size * row_size);

        extern "C" {
            fn risc0_zkp_cuda_sha_rows(
                output: DevicePointer<u8>,
                matrix: DevicePointer<u8>,
                row_size: u32,
                col_size: u32,
            ) -> *const std::os::raw::c_char;
        }

        execute_kernel(|| {
            ffi_wrap(|| unsafe {
                risc0_zkp_cuda_sha_rows(
                    output.as_device_ptr(),
                    matrix.as_device_ptr(),
                    row_size as u32,
                    col_size as u32,
                )
            })
        })
        .unwrap();
    }

    fn get_hash_suite(&self) -> &HashSuite<BabyBear> {
        &self.suite
    }
}

pub struct CudaHashPoseidon2 {
    suite: HashSuite<BabyBear>,
}

impl CudaHash for CudaHashPoseidon2 {
    fn new() -> Self {
        CudaHashPoseidon2 {
            suite: Poseidon2HashSuite::new_suite(),
        }
    }

    fn hash_fold(&self, io: &BufferImpl<Digest>, output_size: usize) {
        // Synchronize our stream before calling sppark to avoid conflicts
        self.stream().synchronize().unwrap();

        let err = unsafe {
            let input = io.as_device_ptr_with_offset(2 * output_size);
            let output = io.as_device_ptr_with_offset(output_size);
            sppark_poseidon2_fold(output, input, output_size)
        };
        if err.code != 0 {
            panic!("Failure during hash_fold: {err}");
        }
    }

    fn hash_rows(&self, output: &BufferImpl<Digest>, matrix: &BufferImpl<BabyBearElem>) {
        // Synchronize our stream before calling sppark to avoid conflicts
        self.stream().synchronize().unwrap();

        let row_size = output.size();
        let col_size = matrix.size() / output.size();
        assert_eq!(matrix.size(), col_size * row_size);

        let err = unsafe {
            sppark_poseidon2_rows(
                output.as_device_ptr(),
                matrix.as_device_ptr(),
                row_size.try_into().unwrap(),
                col_size.try_into().unwrap(),
            )
        };
        if err.code != 0 {
            panic!("Failure during hash_rows: {err}");
        }
    }

    fn get_hash_suite(&self) -> &HashSuite<BabyBear> {
        &self.suite
    }
}

pub struct CudaHashPoseidon254 {
    suite: HashSuite<BabyBear>,
}

impl CudaHash for CudaHashPoseidon254 {
    fn new() -> Self {
        CudaHashPoseidon254 {
            suite: Poseidon254HashSuite::new_suite(),
        }
    }

    fn hash_fold(&self, io: &BufferImpl<Digest>, output_size: usize) {
        // Synchronize our stream before calling sppark to avoid conflicts
        self.stream().synchronize().unwrap();

        let err = unsafe {
            let input = io.as_device_ptr_with_offset(2 * output_size);
            let output = io.as_device_ptr_with_offset(output_size);
            sppark_poseidon254_fold(output, input, output_size)
        };
        if err.code != 0 {
            panic!("Failure during hash_fold: {err}");
        }
    }

    fn hash_rows(&self, output: &BufferImpl<Digest>, matrix: &BufferImpl<BabyBearElem>) {
        // Synchronize our stream before calling sppark to avoid conflicts
        self.stream().synchronize().unwrap();

        let row_size = output.size();
        let col_size = matrix.size() / output.size();
        assert_eq!(matrix.size(), col_size * row_size);

        let err = unsafe {
            sppark_poseidon254_rows(
                output.as_device_ptr(),
                matrix.as_device_ptr(),
                row_size,
                col_size.try_into().unwrap(),
            )
        };
        if err.code != 0 {
            panic!("Failure during hash_rows 254: {err}");
        }
    }

    fn get_hash_suite(&self) -> &HashSuite<BabyBear> {
        &self.suite
    }
}

// Static kernel generation counter - increments after each kernel operation
// Used to track which buffers might have been modified by kernels
static KERNEL_GENERATION: AtomicU64 = AtomicU64::new(1);

// Size threshold for caching - only cache buffers under this size (default: 100MB)
// Large buffers are less likely to benefit from caching due to memory pressure
const CACHE_SIZE_THRESHOLD: usize = 100 * 1024 * 1024;

/// Execute a kernel operation and increment kernel generation counter
/// This ensures buffers used in kernels are marked as potentially modified
///
/// Note: After kernel execution, consider calling `warm_cache_after_kernel()`
/// on output buffers that will be read to pre-populate their cache.
fn execute_kernel<F>(f: F) -> Result<()>
where
    F: FnOnce() -> Result<()>,
{
    let result = f();
    // Increment kernel generation after kernel completes
    // This marks all buffers that had device pointers accessed as potentially modified
    KERNEL_GENERATION.fetch_add(1, Ordering::Relaxed);
    result
}

/// Execute a kernel operation with stream synchronization
/// This version ensures the stream is synchronized after kernel execution
/// Use this when you need to ensure kernel completion before proceeding
fn execute_kernel_with_stream<F>(stream: &Stream, f: F) -> Result<()>
where
    F: FnOnce() -> Result<()>,
{
    let result = f();
    // Synchronize stream to ensure kernel completes before incrementing generation
    stream.synchronize()?;
    // Increment kernel generation after kernel completes
    KERNEL_GENERATION.fetch_add(1, Ordering::Relaxed);
    result
}

pub struct CudaHal<Hash: CudaHash + ?Sized> {
    pub max_threads: u32,
    hash: Option<Box<Hash>>,
    _context: Context,
    stream: Stream,
    // Dedicated stream for sppark operations to isolate them from our main stream
    // Note: sppark still calls cudaDeviceSynchronize() which blocks ALL streams,
    // so this is a workaround until we fork sppark to remove blocking calls
    #[allow(dead_code)]
    sppark_stream: Stream,
    _lock: ReentrantMutexGuard<'static, ()>,
}

pub type CudaHalSha256 = CudaHal<CudaHashSha256>;
pub type CudaHalPoseidon2 = CudaHal<CudaHashPoseidon2>;
pub type CudaHalPoseidon254 = CudaHal<CudaHashPoseidon254>;

struct RawBuffer {
    name: &'static str,
    buf: DeviceBuffer<u8>,
    // Host-side cache to avoid repeated D2H transfers
    // This provides significant speedup for frequently accessed buffers
    cached_host: Option<Vec<u8>>,
    // Kernel generation when this buffer's device pointer was last accessed
    // If this >= cache_kernel_gen, buffer might have been modified by a kernel
    last_kernel_gen: std::cell::Cell<u64>,
    // Kernel generation when cache was created
    // Cache is valid only if last_kernel_gen < cache_kernel_gen
    cache_kernel_gen: std::cell::Cell<u64>,
    // Access count - tracks how many times this buffer has been accessed
    // Used to prioritize caching frequently accessed buffers
    access_count: std::cell::Cell<u32>,
}

impl RawBuffer {
    pub fn new(name: &'static str, size: usize) -> Self {
        tracing::trace!("alloc: {size} bytes, {name}");
        tracker().lock().unwrap().alloc(size);
        let buf = unsafe { DeviceBuffer::uninitialized(size) }
            .context(format!("allocation failed on {name}: {size} bytes"))
            .unwrap();
        Self {
            name,
            buf,
            cached_host: None,
            last_kernel_gen: std::cell::Cell::new(0),
            cache_kernel_gen: std::cell::Cell::new(0),
            access_count: std::cell::Cell::new(0),
        }
    }

    pub fn set_u32(&mut self, value: u32) {
        self.buf.set_32(value).unwrap();
        // Lazy cache invalidation - only clear if cache exists
        // This avoids unnecessary work if cache wasn't populated
        if self.cached_host.is_some() {
            self.cached_host = None;
        }
    }

    /// Get host-side copy, using cache if available and up-to-date.
    /// Cache is only used if the buffer hasn't been used in any kernel since cache was created.
    fn get_host_copy(&mut self) -> &[u8] {
        // Track access for prioritization
        self.access_count
            .set(self.access_count.get().saturating_add(1));

        // Fast path: if buffer was never used in kernel (last_gen == 0), cache is always valid
        let last_gen = self.last_kernel_gen.get();

        if last_gen == 0 {
            // Buffer never used in kernel - use cache if available
            if self.cached_host.is_none() {
                // Cache the data (even for large buffers, we need to store it to return a reference)
                // For large buffers, this cache may be evicted more aggressively in the future
                self.cached_host = Some(self.buf.as_host_vec().unwrap());
            }
            return self.cached_host.as_ref().unwrap();
        }

        // Buffer was used in kernel - need to check if cache is still valid
        // Only do expensive atomic load if we have a cache to check
        if let Some(_) = &self.cached_host {
            let cache_gen = self.cache_kernel_gen.get();
            // If buffer was used before cache was created, cache is still valid
            if last_gen < cache_gen {
                return self.cached_host.as_ref().unwrap();
            }
        }

        // Cache invalid or doesn't exist - fetch fresh from device
        let current_gen = KERNEL_GENERATION.load(Ordering::Relaxed);
        // Cache the data (we need to store it to return a reference)
        // For large or rarely accessed buffers, we cache but may evict more aggressively
        self.cached_host = Some(self.buf.as_host_vec().unwrap());
        self.cache_kernel_gen.set(current_gen);
        self.last_kernel_gen.set(0); // Reset since we just synced
        self.cached_host.as_ref().unwrap()
    }

    /// Prefetch hint - indicates this buffer will be accessed soon
    /// This allows us to start fetching data asynchronously if possible
    fn prefetch_hint(&mut self) {
        // If buffer is likely to be accessed and cache is missing, start prefetching
        // For now, we just ensure cache exists if buffer hasn't been used in kernel
        let last_gen = self.last_kernel_gen.get();
        if last_gen == 0 && self.cached_host.is_none() && self.buf.len() <= CACHE_SIZE_THRESHOLD {
            // Pre-populate cache for buffers that haven't been used in kernels
            self.cached_host = Some(self.buf.as_host_vec().unwrap());
        }
    }

    /// Warm cache for a buffer that will be read after kernel execution
    /// Call this after kernel operations to pre-populate cache for output buffers
    fn warm_cache_after_kernel(&mut self) {
        let current_gen = KERNEL_GENERATION.load(Ordering::Relaxed);
        let last_gen = self.last_kernel_gen.get();

        // Only warm cache if:
        // 1. Buffer was used in kernel (last_gen > 0)
        // 2. Buffer is under size threshold
        // 3. Cache doesn't exist or is invalid
        if last_gen > 0
            && self.buf.len() <= CACHE_SIZE_THRESHOLD
            && (self.cached_host.is_none() || last_gen >= self.cache_kernel_gen.get())
        {
            self.cached_host = Some(self.buf.as_host_vec().unwrap());
            self.cache_kernel_gen.set(current_gen);
            self.last_kernel_gen.set(0); // Reset since we just synced
        }
    }

    /// Mark that this buffer's device pointer was accessed (might be used in kernel)
    fn mark_device_ptr_accessed(&mut self) {
        // Only update if not already marked (avoid atomic load if already marked)
        if self.last_kernel_gen.get() == 0 {
            let current_gen = KERNEL_GENERATION.load(Ordering::Relaxed);
            self.last_kernel_gen.set(current_gen);
        }
    }

    /// Set memory advice hint: mark buffer as read-mostly (optimized for GPU reads)
    /// Use this for input buffers that are primarily read by kernels
    fn advise_read_mostly(&self) {
        let ptr = self.buf.as_device_ptr().as_raw() as *const std::os::raw::c_void;
        let size = self.buf.len();
        unsafe {
            let result = cudaMemAdvise(
                ptr,
                size,
                CudaMemAdvise::SetReadMostly as u32,
                0, // Current device
            );
            // Ignore errors - memory advice is just a hint, CUDA may ignore it
            if result != 0 {
                tracing::trace!("cudaMemAdvise SetReadMostly returned {}", result);
            }
        }
    }

    /// Set memory advice hint: mark buffer with preferred location on GPU
    /// Use this for output buffers that will be written by kernels
    fn advise_preferred_location_gpu(&self) {
        let ptr = self.buf.as_device_ptr().as_raw() as *const std::os::raw::c_void;
        let size = self.buf.len();
        unsafe {
            let result = cudaMemAdvise(
                ptr,
                size,
                CudaMemAdvise::SetPreferredLocation as u32,
                0, // Current device
            );
            // Ignore errors - memory advice is just a hint
            if result != 0 {
                tracing::trace!("cudaMemAdvise SetPreferredLocation returned {}", result);
            }
        }
    }

    /// Update device buffer from host cache (for view_mut)
    fn sync_to_device(&mut self, host_data: &[u8]) {
        self.buf.copy_from(host_data).unwrap();
        // After sync, cache is valid again and buffer hasn't been used in kernel
        // No need to load atomic - just reset to 0 (never used in kernel)
        self.last_kernel_gen.set(0);
        // Update cache without needing to track generation since last_gen is 0
        if let Some(ref mut cached) = self.cached_host {
            cached.copy_from_slice(host_data);
        }
    }

    /// Async copy from host to device using a CUDA stream
    /// Note: The stream handle needs to be obtained from cust::Stream
    /// For now, we use synchronous copy but this provides the interface for async
    #[allow(dead_code)]
    fn copy_from_async(&mut self, _stream: &Stream, host_data: &[u8]) -> Result<()> {
        // TODO: Use cudaMemcpyAsync once we can get raw stream handle from cust::Stream
        // For now, fall back to synchronous copy
        self.buf.copy_from(host_data)?;

        // Update cache
        self.last_kernel_gen.set(0);
        if let Some(ref mut cached) = self.cached_host {
            cached.copy_from_slice(host_data);
        }

        Ok(())
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        tracing::trace!("free: {} bytes, {}", self.buf.len(), self.name);
        tracker().lock().unwrap().free(self.buf.len());
    }
}

#[derive(Clone)]
pub struct BufferImpl<T> {
    buffer: Rc<RefCell<RawBuffer>>,
    size: usize,
    offset: usize,
    marker: PhantomData<T>,
}

#[inline]
fn unchecked_cast<A, B>(a: &[A]) -> &[B] {
    let new_len = std::mem::size_of_val(a) / std::mem::size_of::<B>();
    unsafe { std::slice::from_raw_parts(a.as_ptr() as *const B, new_len) }
}

#[inline]
fn unchecked_cast_mut<A, B>(a: &mut [A]) -> &mut [B] {
    let new_len = std::mem::size_of_val(a) / std::mem::size_of::<B>();
    unsafe { std::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut B, new_len) }
}

impl<T> BufferImpl<T> {
    fn new(name: &'static str, size: usize) -> Self {
        let bytes_len = std::mem::size_of::<T>() * size;
        assert!(bytes_len > 0);
        BufferImpl {
            buffer: Rc::new(RefCell::new(RawBuffer::new(name, bytes_len))),
            size,
            offset: 0,
            marker: PhantomData,
        }
    }

    pub fn copy_from(name: &'static str, slice: &[T]) -> Self {
        // scope!("copy_from");
        let bytes_len = std::mem::size_of_val(slice);
        assert!(bytes_len > 0);
        let mut buffer = RawBuffer::new(name, bytes_len);
        let bytes = unchecked_cast(slice);
        buffer.buf.copy_from(bytes).unwrap();
        // Populate cache immediately since we just copied from host
        // This avoids D2H transfer on first access
        // No need to set cache_kernel_gen since last_kernel_gen is 0 (never used in kernel)
        buffer.cached_host = Some(bytes.to_vec());
        buffer.last_kernel_gen.set(0); // Buffer hasn't been used in kernel yet

        BufferImpl {
            buffer: Rc::new(RefCell::new(buffer)),
            size: slice.len(),
            offset: 0,
            marker: PhantomData,
        }
    }

    pub fn as_device_ptr(&self) -> DevicePointer<u8> {
        // Mark that this buffer's device pointer is being accessed (might be used in kernel)
        // Do this in a single borrow to avoid double borrow overhead
        let mut buf = self.buffer.borrow_mut();
        buf.mark_device_ptr_accessed();
        let ptr = buf.buf.as_device_ptr();
        let offset = self.offset * std::mem::size_of::<T>();
        unsafe { ptr.offset(offset.try_into().unwrap()) }
    }

    /// Mark this buffer as read-mostly (optimized for GPU reads)
    /// Call this for input buffers that are primarily read by kernels
    pub fn mark_read_mostly(&self) {
        self.buffer.borrow().advise_read_mostly();
    }

    /// Mark this buffer with preferred location on GPU
    /// Call this for output buffers that will be written by kernels
    pub fn mark_preferred_location_gpu(&self) {
        self.buffer.borrow().advise_preferred_location_gpu();
    }

    pub fn as_device_ptr_with_offset(&self, offset: usize) -> DevicePointer<u8> {
        // Mark that this buffer's device pointer is being accessed (might be used in kernel)
        // Do this in a single borrow to avoid double borrow overhead
        let mut buf = self.buffer.borrow_mut();
        buf.mark_device_ptr_accessed();
        let ptr = buf.buf.as_device_ptr();
        let offset = (self.offset + offset) * std::mem::size_of::<T>();
        unsafe { ptr.offset(offset.try_into().unwrap()) }
    }

    /// Hint that this buffer will be accessed soon - allows prefetching
    pub fn prefetch_hint(&self) {
        self.buffer.borrow_mut().prefetch_hint();
    }

    /// Warm cache after kernel execution - call this for output buffers that will be read
    pub fn warm_cache_after_kernel(&self) {
        self.buffer.borrow_mut().warm_cache_after_kernel();
    }
}

impl<T: Clone> Buffer<T> for BufferImpl<T> {
    fn name(&self) -> &'static str {
        self.buffer.borrow().name
    }

    fn size(&self) -> usize {
        self.size
    }

    fn slice(&self, offset: usize, size: usize) -> BufferImpl<T> {
        assert!(offset + size <= self.size());
        BufferImpl {
            buffer: self.buffer.clone(),
            size,
            offset: self.offset + offset,
            marker: PhantomData,
        }
    }

    fn get_at(&self, idx: usize) -> T {
        // Use cached host copy if available to avoid D2H transfer
        let mut buf = self.buffer.borrow_mut();
        let host_copy = buf.get_host_copy();
        let item_size = std::mem::size_of::<T>();
        let offset = (self.offset + idx) * item_size;
        let slice: &[T] = unchecked_cast(&host_copy[offset..offset + item_size]);
        slice[0].clone()
    }

    fn view<F: FnOnce(&[T])>(&self, f: F) {
        scope!("view");
        // Use cached host copy to avoid D2H transfer
        let mut buf = self.buffer.borrow_mut();
        let host_copy = buf.get_host_copy();
        let item_size = std::mem::size_of::<T>();
        let offset = self.offset * item_size;
        let len = self.size * item_size;
        let slice = unchecked_cast(&host_copy[offset..offset + len]);
        f(slice);
    }

    fn view_mut<F: FnOnce(&mut [T])>(&self, f: F) {
        scope!("view_mut");
        let mut buf = self.buffer.borrow_mut();
        // Get or create host copy
        let host_copy = buf.get_host_copy().to_vec();
        let mut host_buf = host_copy;
        let item_size = std::mem::size_of::<T>();
        let offset = self.offset * item_size;
        let len = self.size * item_size;
        let slice = unchecked_cast_mut(&mut host_buf[offset..offset + len]);
        f(slice);
        // Sync back to device
        buf.sync_to_device(&host_buf);
    }

    fn to_vec(&self) -> Vec<T> {
        // Use cached host copy to avoid D2H transfer
        let mut buf = self.buffer.borrow_mut();
        let host_copy = buf.get_host_copy();
        let item_size = std::mem::size_of::<T>();
        let offset = self.offset * item_size;
        let len = self.size * item_size;
        let slice = unchecked_cast(&host_copy[offset..offset + len]);
        slice.to_vec()
    }
}

impl<CH: CudaHash> Default for CudaHal<CH> {
    fn default() -> Self {
        Self::new()
    }
}

impl<CH: CudaHash + ?Sized> CudaHal<CH> {
    pub fn new() -> Self
    where
        CH: Sized,
    {
        Self::new_from_hash(Box::new(CH::new()))
    }

    /// Get the CUDA stream for async operations
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    fn new_from_hash(hash: Box<CH>) -> Self {
        let _lock = singleton().lock();

        let err = unsafe { sppark_init() };
        if err.code != 0 {
            panic!("Failure during sppark_init: {err}");
        }

        cust::init(CudaFlags::empty()).unwrap();
        let device = Device::get_device(0).unwrap();
        let max_threads = device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap();
        let context = Context::new(device).unwrap();
        context.set_flags(ContextFlags::SCHED_AUTO).unwrap();

        // Create a CUDA stream for async operations
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .context("Failed to create CUDA stream")
            .unwrap();

        // Create a dedicated stream for sppark operations
        // This isolates sppark from our main stream, though sppark's internal
        // cudaDeviceSynchronize() calls will still block all streams
        let sppark_stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .context("Failed to create sppark CUDA stream")
            .unwrap();

        let mut hal = Self {
            max_threads: max_threads as u32,
            _context: context,
            stream,
            sppark_stream,
            hash: None,
            _lock,
        };
        hal.hash = Some(hash);
        hal
    }

    fn poly_divide(
        &self,
        polynomial: &BufferImpl<BabyBearExtElem>,
        pow: BabyBearExtElem,
    ) -> BabyBearExtElem {
        let mut remainder = BabyBearExtElem::ZERO;
        let poly_size = polynomial.size();
        let pow = pow.to_u32_words();

        let err = unsafe {
            supra_poly_divide(
                polynomial.as_device_ptr(),
                poly_size,
                &mut remainder as *mut _ as *mut u32,
                pow.as_ptr(),
            )
        };

        if err.code != 0 {
            panic!("Failure during supra_poly_divide: {err}");
        }

        remainder
    }
}

impl CudaHal<dyn CudaHash> {
    pub fn new_from_hash_suite(hash_suite: HashSuite<BabyBear>) -> Result<Self> {
        let hash_suite_box = match &hash_suite.name[..] {
            "poseidon2" => Box::new(CudaHashPoseidon2::new()) as Box<dyn CudaHash>,
            "poseidon254" => Box::new(CudaHashPoseidon254::new()) as Box<dyn CudaHash>,
            "sha-256" => Box::new(CudaHashSha256::new()) as Box<dyn CudaHash>,
            other => bail!("unsupported hash_fn {other}"),
        };
        Ok(Self::new_from_hash(hash_suite_box))
    }
}

impl<CH: CudaHash + ?Sized> Hal for CudaHal<CH> {
    type Field = BabyBear;
    type Elem = BabyBearElem;
    type ExtElem = BabyBearExtElem;
    type Buffer<T: Clone + Debug + PartialEq> = BufferImpl<T>;

    fn alloc_elem(&self, name: &'static str, size: usize) -> Self::Buffer<Self::Elem> {
        let buffer = BufferImpl::new(name, size);
        // Mark newly allocated buffers as preferred location on GPU (they'll be written to)
        buffer.mark_preferred_location_gpu();
        buffer
    }

    fn alloc_elem_init(
        &self,
        name: &'static str,
        size: usize,
        value: Self::Elem,
    ) -> Self::Buffer<Self::Elem> {
        let buffer = self.alloc_elem(name, size);
        buffer
            .buffer
            .borrow_mut()
            .set_u32(value.as_u32_montgomery());
        buffer
    }

    fn copy_from_elem(&self, name: &'static str, slice: &[Self::Elem]) -> Self::Buffer<Self::Elem> {
        let buffer = BufferImpl::copy_from(name, slice);
        // Mark input buffers (copied from host) as read-mostly (they'll be read by kernels)
        buffer.mark_read_mostly();
        // TODO: Use async copy with stream for better performance
        // For now, synchronous copy is used but stream is available via self.stream()
        buffer
    }

    fn alloc_extelem(&self, name: &'static str, size: usize) -> Self::Buffer<Self::ExtElem> {
        let buffer = BufferImpl::new(name, size);
        // Mark newly allocated buffers as preferred location on GPU
        buffer.mark_preferred_location_gpu();
        buffer
    }

    fn alloc_extelem_zeroed(&self, name: &'static str, size: usize) -> Self::Buffer<Self::ExtElem> {
        let buffer = self.alloc_extelem(name, size);
        buffer.buffer.borrow_mut().set_u32(0);
        buffer
    }

    fn copy_from_extelem(
        &self,
        name: &'static str,
        slice: &[Self::ExtElem],
    ) -> Self::Buffer<Self::ExtElem> {
        let buffer = BufferImpl::copy_from(name, slice);
        // Mark input buffers as read-mostly
        buffer.mark_read_mostly();
        buffer
    }

    fn alloc_digest(&self, name: &'static str, size: usize) -> Self::Buffer<Digest> {
        let buffer = BufferImpl::new(name, size);
        // Mark newly allocated buffers as preferred location on GPU
        buffer.mark_preferred_location_gpu();
        buffer
    }

    fn copy_from_digest(&self, name: &'static str, slice: &[Digest]) -> Self::Buffer<Digest> {
        let buffer = BufferImpl::copy_from(name, slice);
        // Mark input buffers as read-mostly
        buffer.mark_read_mostly();
        buffer
    }

    fn alloc_u32(&self, name: &'static str, size: usize) -> Self::Buffer<u32> {
        let buffer = BufferImpl::new(name, size);
        // Mark newly allocated buffers as preferred location on GPU
        buffer.mark_preferred_location_gpu();
        buffer
    }

    fn copy_from_u32(&self, name: &'static str, slice: &[u32]) -> Self::Buffer<u32> {
        let buffer = BufferImpl::copy_from(name, slice);
        // Mark input buffers as read-mostly
        buffer.mark_read_mostly();
        buffer
    }

    fn batch_expand_into_evaluate_ntt(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::Elem>,
        poly_count: usize,
        expand_bits: usize,
    ) {
        // Synchronize our stream before calling sppark to avoid conflicts
        self.stream().synchronize().unwrap();

        // batch_expand
        {
            let out_size = output.size() / poly_count;
            let in_size = input.size() / poly_count;
            let expand_bits = log2_ceil(out_size / in_size);
            assert_eq!(output.size(), out_size * poly_count);
            assert_eq!(input.size(), in_size * poly_count);
            assert_eq!(out_size, in_size * (1 << expand_bits));
            let in_bits = log2_ceil(in_size);
            let err = unsafe {
                sppark_batch_expand(
                    output.as_device_ptr(),
                    input.as_device_ptr(),
                    in_bits.try_into().unwrap(),
                    expand_bits.try_into().unwrap(),
                    poly_count.try_into().unwrap(),
                )
            };
            if err.code != 0 {
                panic!("Failure during batch_expand: {err}");
            }
        }

        // batch_evaluate_ntt
        {
            let row_size = output.size() / poly_count;
            assert_eq!(row_size * poly_count, output.size());
            let n_bits = log2_ceil(row_size);
            assert_eq!(row_size, 1 << n_bits);
            assert!(n_bits >= expand_bits);
            assert!(n_bits < Self::Elem::MAX_ROU_PO2);

            let err = unsafe {
                sppark_batch_NTT(
                    output.as_device_ptr(),
                    n_bits.try_into().unwrap(),
                    poly_count.try_into().unwrap(),
                )
            };
            if err.code != 0 {
                panic!("Failure during batch_evaluate_ntt: {err}");
            }
        }
    }

    fn batch_interpolate_ntt(&self, io: &Self::Buffer<Self::Elem>, count: usize) {
        // Synchronize our stream before calling sppark to avoid conflicts
        self.stream().synchronize().unwrap();

        let row_size = io.size() / count;
        assert_eq!(row_size * count, io.size());
        let n_bits = log2_ceil(row_size);
        assert_eq!(row_size, 1 << n_bits);
        assert!(n_bits < Self::Elem::MAX_ROU_PO2);

        let err = unsafe {
            sppark_batch_iNTT(
                io.as_device_ptr(),
                n_bits.try_into().unwrap(),
                count.try_into().unwrap(),
            )
        };
        if err.code != 0 {
            panic!("Failure during batch_interpolate_ntt: {err}");
        }
    }

    fn batch_bit_reverse(&self, io: &Self::Buffer<Self::Elem>, count: usize) {
        let row_size = io.size() / count;
        assert_eq!(row_size * count, io.size());
        let bits = log2_ceil(row_size);
        assert_eq!(row_size, 1 << bits);
        let io_size = io.size();

        extern "C" {
            fn risc0_zkp_cuda_batch_bit_reverse(
                io: DevicePointer<u8>,
                bits: u32,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        // Use stream for async kernel execution
        execute_kernel_with_stream(self.stream(), || {
            ffi_wrap(|| unsafe {
                risc0_zkp_cuda_batch_bit_reverse(io.as_device_ptr(), bits as u32, io_size as u32)
            })
        })
        .unwrap();
    }

    fn batch_evaluate_any(
        &self,
        coeffs: &Self::Buffer<Self::Elem>,
        poly_count: usize,
        which: &Self::Buffer<u32>,
        xs: &Self::Buffer<Self::ExtElem>,
        out: &Self::Buffer<Self::ExtElem>,
    ) {
        let po2 = log2_ceil(coeffs.size() / poly_count);
        let count = 1 << po2;
        assert_eq!(poly_count * count, coeffs.size());
        let eval_count = which.size();
        assert_eq!(xs.size(), eval_count);
        assert_eq!(out.size(), eval_count);

        let threads_per_block = self.max_threads / 4;
        const BYTES_PER_WORD: u32 = 4;
        const WORDS_PER_FPEXT: u32 = 4;
        let shared_size = threads_per_block * BYTES_PER_WORD * WORDS_PER_FPEXT;
        let kernel_count = out.size() * threads_per_block as usize;

        extern "C" {
            fn risc0_zkp_cuda_batch_evaluate_any(
                output: DevicePointer<u8>,
                coeffs: DevicePointer<u8>,
                which: DevicePointer<u8>,
                xs: DevicePointer<u8>,
                shared_size: u32,
                kernel_count: u32,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        // Use stream for async kernel execution
        execute_kernel_with_stream(self.stream(), || {
            ffi_wrap(|| unsafe {
                risc0_zkp_cuda_batch_evaluate_any(
                    out.as_device_ptr(),
                    coeffs.as_device_ptr(),
                    which.as_device_ptr(),
                    xs.as_device_ptr(),
                    shared_size,
                    kernel_count as u32,
                    count as u32,
                )
            })
        })
        .unwrap();
    }

    fn gather_sample(
        &self,
        dst: &Self::Buffer<Self::Elem>,
        src: &Self::Buffer<Self::Elem>,
        idx: usize,
        size: usize,
        stride: usize,
    ) {
        extern "C" {
            fn risc0_zkp_cuda_gather_sample(
                dst: DevicePointer<u8>,
                src: DevicePointer<u8>,
                idx: u32,
                size: u32,
                stride: u32,
            ) -> *const std::os::raw::c_char;
        }

        // Use stream for async kernel execution
        execute_kernel_with_stream(self.stream(), || {
            ffi_wrap(|| unsafe {
                risc0_zkp_cuda_gather_sample(
                    dst.as_device_ptr(),
                    src.as_device_ptr(),
                    idx as u32,
                    size as u32,
                    stride as u32,
                )
            })
        })
        .unwrap();
    }

    fn has_unified_memory(&self) -> bool {
        false
    }

    fn zk_shift(&self, io: &Self::Buffer<Self::Elem>, poly_count: usize) {
        let bits = log2_ceil(io.size() / poly_count);
        assert_eq!(io.size(), poly_count * (1 << bits));

        // Synchronize our stream before calling sppark to avoid conflicts
        // sppark calls cudaDeviceSynchronize() which blocks ALL streams,
        // so we need to ensure our stream operations complete first
        self.stream().synchronize().unwrap();

        // Note: sppark library manages its own streams and calls cudaDeviceSynchronize()
        // which blocks all streams, so we don't use our stream here
        execute_kernel(|| {
            let err = unsafe {
                sppark_batch_zk_shift(
                    io.as_device_ptr(),
                    bits.try_into().unwrap(),
                    poly_count.try_into().unwrap(),
                )
            };
            if err.code != 0 {
                panic!("Failure during zk_shift: {err}");
            }
            Ok(())
        })
        .unwrap();
    }

    fn mix_poly_coeffs(
        &self,
        output: &Self::Buffer<Self::ExtElem>,
        mix_start: &Self::ExtElem,
        mix: &Self::ExtElem,
        input: &Self::Buffer<Self::Elem>,
        combos: &Self::Buffer<u32>,
        input_size: usize,
        count: usize,
    ) {
        let mix_start = self.copy_from_extelem("mix_start", &[*mix_start]);
        let mix = self.copy_from_extelem("mix", &[*mix]);

        extern "C" {
            fn risc0_zkp_cuda_mix_poly_coeffs(
                output: DevicePointer<u8>,
                input: DevicePointer<u8>,
                combos: DevicePointer<u8>,
                mix_start: DevicePointer<u8>,
                mix: DevicePointer<u8>,
                input_size: u32,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        // Use stream for async kernel execution
        execute_kernel_with_stream(self.stream(), || {
            ffi_wrap(|| unsafe {
                risc0_zkp_cuda_mix_poly_coeffs(
                    output.as_device_ptr(),
                    input.as_device_ptr(),
                    combos.as_device_ptr(),
                    mix_start.as_device_ptr(),
                    mix.as_device_ptr(),
                    input_size as u32,
                    count as u32,
                )
            })
        })
        .unwrap();
    }

    fn eltwise_add_elem(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input1: &Self::Buffer<Self::Elem>,
        input2: &Self::Buffer<Self::Elem>,
    ) {
        assert_eq!(output.size(), input1.size());
        assert_eq!(output.size(), input2.size());
        let count = output.size();

        extern "C" {
            fn risc0_zkp_cuda_eltwise_add_fp(
                out: DevicePointer<u8>,
                x: DevicePointer<u8>,
                y: DevicePointer<u8>,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        // Use stream for async kernel execution
        execute_kernel_with_stream(self.stream(), || {
            ffi_wrap(|| unsafe {
                risc0_zkp_cuda_eltwise_add_fp(
                    output.as_device_ptr(),
                    input1.as_device_ptr(),
                    input2.as_device_ptr(),
                    count as u32,
                )
            })
        })
        .unwrap();
    }

    fn eltwise_sum_extelem(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::ExtElem>,
    ) {
        let count = output.size() / Self::ExtElem::EXT_SIZE;
        let to_add = input.size() / count;
        assert_eq!(output.size(), count * Self::ExtElem::EXT_SIZE);
        assert_eq!(input.size(), count * to_add);

        extern "C" {
            fn risc0_zkp_cuda_eltwise_sum_fpext(
                output: DevicePointer<u8>,
                input: DevicePointer<u8>,
                to_add: u32,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        // Use stream for async kernel execution
        execute_kernel_with_stream(self.stream(), || {
            ffi_wrap(|| unsafe {
                risc0_zkp_cuda_eltwise_sum_fpext(
                    output.as_device_ptr(),
                    input.as_device_ptr(),
                    to_add as u32,
                    count as u32,
                )
            })
        })
        .unwrap();
    }

    fn eltwise_copy_elem(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::Elem>,
    ) {
        let count = output.size();
        assert_eq!(count, input.size());

        extern "C" {
            fn risc0_zkp_cuda_eltwise_copy_fp(
                output: DevicePointer<u8>,
                input: DevicePointer<u8>,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        // Use stream for async kernel execution
        execute_kernel_with_stream(self.stream(), || {
            ffi_wrap(|| unsafe {
                risc0_zkp_cuda_eltwise_copy_fp(
                    output.as_device_ptr(),
                    input.as_device_ptr(),
                    count as u32,
                )
            })
        })
        .unwrap();
    }

    fn eltwise_zeroize_elem(&self, elems: &Self::Buffer<Self::Elem>) {
        extern "C" {
            fn risc0_zkp_cuda_eltwise_zeroize_fp(
                elems: DevicePointer<u8>,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        // Use stream for async kernel execution
        execute_kernel_with_stream(self.stream(), || {
            ffi_wrap(|| unsafe {
                risc0_zkp_cuda_eltwise_zeroize_fp(elems.as_device_ptr(), elems.size() as u32)
            })
        })
        .unwrap();
    }

    fn scatter(
        &self,
        into: &Self::Buffer<Self::Elem>,
        index: &[u32],
        offsets: &[u32],
        values: &[Self::Elem],
    ) {
        if index.is_empty() {
            return;
        }

        let count = index.len() - 1;
        if count == 0 {
            return;
        }

        let index = self.copy_from_u32("index", index);
        let offsets = self.copy_from_u32("offsets", offsets);
        let values = self.copy_from_elem("values", values);

        extern "C" {
            fn risc0_zkp_cuda_scatter(
                into: DevicePointer<u8>,
                index: DevicePointer<u8>,
                offsets: DevicePointer<u8>,
                values: DevicePointer<u8>,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        // Use stream for async kernel execution
        execute_kernel_with_stream(self.stream(), || {
            ffi_wrap(|| unsafe {
                risc0_zkp_cuda_scatter(
                    into.as_device_ptr(),
                    index.as_device_ptr(),
                    offsets.as_device_ptr(),
                    values.as_device_ptr(),
                    count as u32,
                )
            })
        })
        .unwrap();
    }

    fn eltwise_copy_elem_slice(
        &self,
        into: &Self::Buffer<Self::Elem>,
        from: &[Self::Elem],
        from_rows: usize,
        from_cols: usize,
        from_offset: usize,
        from_stride: usize,
        into_offset: usize,
        into_stride: usize,
    ) {
        let from = self.copy_from_elem("from", from);

        extern "C" {
            fn risc0_zkp_cuda_eltwise_copy_fp_region(
                into: DevicePointer<u8>,
                from: DevicePointer<u8>,
                from_rows: u32,
                from_cols: u32,
                from_offset: u32,
                from_stride: u32,
                into_offset: u32,
                into_stride: u32,
            ) -> *const std::os::raw::c_char;
        }

        // Use stream for async kernel execution
        execute_kernel_with_stream(self.stream(), || {
            ffi_wrap(|| unsafe {
                risc0_zkp_cuda_eltwise_copy_fp_region(
                    into.as_device_ptr(),
                    from.as_device_ptr(),
                    from_rows as u32,
                    from_cols as u32,
                    from_offset as u32,
                    from_stride as u32,
                    into_offset as u32,
                    into_stride as u32,
                )
            })
        })
        .unwrap();
    }

    fn fri_fold(
        &self,
        output: &Self::Buffer<Self::Elem>,
        input: &Self::Buffer<Self::Elem>,
        mix: &Self::ExtElem,
    ) {
        let count = output.size() / Self::ExtElem::EXT_SIZE;
        assert_eq!(output.size(), count * Self::ExtElem::EXT_SIZE);
        assert_eq!(input.size(), output.size() * FRI_FOLD);
        let mix = self.copy_from_extelem("mix", &[*mix]);

        extern "C" {
            fn risc0_zkp_cuda_fri_fold(
                output: DevicePointer<u8>,
                input: DevicePointer<u8>,
                mix: DevicePointer<u8>,
                count: u32,
            ) -> *const std::os::raw::c_char;
        }

        // Use stream for async kernel execution
        execute_kernel_with_stream(self.stream(), || {
            ffi_wrap(|| unsafe {
                risc0_zkp_cuda_fri_fold(
                    output.as_device_ptr(),
                    input.as_device_ptr(),
                    mix.as_device_ptr(),
                    count as u32,
                )
            })
        })
        .unwrap();
    }

    fn hash_fold(&self, io: &Self::Buffer<Digest>, input_size: usize, output_size: usize) {
        assert_eq!(input_size, 2 * output_size);
        self.hash.as_ref().unwrap().hash_fold(io, output_size);
    }

    fn hash_rows(&self, output: &Self::Buffer<Digest>, matrix: &Self::Buffer<Self::Elem>) {
        self.hash.as_ref().unwrap().hash_rows(output, matrix);
    }

    fn get_hash_suite(&self) -> &HashSuite<Self::Field> {
        self.hash.as_ref().unwrap().get_hash_suite()
    }

    fn prefix_products(&self, io: &Self::Buffer<Self::ExtElem>) {
        io.view_mut(|io| {
            for i in 1..io.len() {
                io[i] *= io[i - 1];
            }
        });
    }

    fn combos_prepare(
        &self,
        combos: &Self::Buffer<Self::ExtElem>,
        coeff_u: &[Self::ExtElem],
        combo_count: usize,
        cycles: usize,
        reg_sizes: &[u32],
        reg_combo_ids: &[u32],
        mix: &Self::ExtElem,
    ) {
        let coeff_u = self.copy_from_extelem("coeff_u", coeff_u);
        let combo_count = combo_count as u32;
        let cycles = cycles as u32;
        let regs_count = reg_sizes.len() as u32;
        let reg_sizes = self.copy_from_u32("reg_sizes", reg_sizes);
        let reg_combo_ids = self.copy_from_u32("reg_combo_ids", reg_combo_ids);
        let mix = self.copy_from_extelem("mix", &[*mix]);

        extern "C" {
            fn risc0_zkp_cuda_combos_prepare(
                combos: DevicePointer<u8>,
                coeff_u: DevicePointer<u8>,
                combo_count: u32,
                cycles: u32,
                regs_count: u32,
                reg_sizes: DevicePointer<u8>,
                reg_combo_ids: DevicePointer<u8>,
                checkSize: u32,
                mix: DevicePointer<u8>,
            ) -> *const std::os::raw::c_char;
        }

        // Use stream for async kernel execution
        execute_kernel_with_stream(self.stream(), || {
            ffi_wrap(|| unsafe {
                risc0_zkp_cuda_combos_prepare(
                    combos.as_device_ptr(),
                    coeff_u.as_device_ptr(),
                    combo_count,
                    cycles,
                    regs_count,
                    reg_sizes.as_device_ptr(),
                    reg_combo_ids.as_device_ptr(),
                    Self::CHECK_SIZE as u32,
                    mix.as_device_ptr(),
                )
            })
        })
        .unwrap();
    }

    fn combos_divide(
        &self,
        combos: &Self::Buffer<Self::ExtElem>,
        chunks: Vec<(usize, Vec<Self::ExtElem>)>,
        cycles: usize,
    ) {
        scope!("combos_divide");
        for (i, pows) in chunks {
            let combo_slice = combos.slice(i * cycles, cycles);
            for pow in pows {
                let remainder = self.poly_divide(&combo_slice, pow);
                assert_eq!(remainder, Self::ExtElem::ZERO, "i: {i}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use test_log::test;

    use super::{CudaHalPoseidon2, CudaHalSha256};
    use crate::hal::testutil;

    #[test]
    #[should_panic]
    fn check_req() {
        testutil::check_req(CudaHalSha256::new());
    }

    #[test]
    fn eltwise_add_elem() {
        testutil::eltwise_add_elem(CudaHalSha256::new());
    }

    #[test]
    fn eltwise_copy_elem() {
        testutil::eltwise_copy_elem(CudaHalSha256::new());
    }

    #[test]
    fn eltwise_sum_extelem() {
        testutil::eltwise_sum_extelem(CudaHalSha256::new());
    }

    #[test]
    fn hash_rows_sha256() {
        testutil::hash_rows(CudaHalSha256::new());
    }

    #[test]
    fn hash_fold_sha256() {
        testutil::hash_fold(CudaHalSha256::new());
    }

    #[test]
    fn hash_rows_poseidon2() {
        testutil::hash_rows(CudaHalPoseidon2::new());
    }

    #[test]
    fn hash_fold_poseidon2() {
        testutil::hash_fold(CudaHalPoseidon2::new());
    }

    #[test]
    fn fri_fold() {
        testutil::fri_fold(CudaHalSha256::new());
    }

    #[test]
    fn batch_expand_into_evaluate_ntt() {
        testutil::batch_expand_into_evaluate_ntt(CudaHalSha256::new());
    }

    #[test]
    fn batch_interpolate_ntt() {
        testutil::batch_interpolate_ntt(CudaHalSha256::new());
    }

    #[test]
    fn batch_bit_reverse() {
        testutil::batch_bit_reverse(CudaHalSha256::new());
    }

    #[test]
    fn batch_evaluate_any() {
        testutil::batch_evaluate_any(CudaHalSha256::new());
    }

    #[test]
    fn gather_sample() {
        testutil::gather_sample(CudaHalSha256::new());
    }

    #[test]
    fn zk_shift() {
        testutil::zk_shift(CudaHalSha256::new());
    }

    #[test]
    fn mix_poly_coeffs() {
        testutil::mix_poly_coeffs(CudaHalSha256::new());
    }
}
