# Advanced CUDA Optimizations for GB10 Performance

## Current State Analysis

We're currently using:
- **DeviceBuffer** (not unified memory) - `has_unified_memory()` returns `false`
- **Synchronous operations** - All memory transfers block the CPU
- **Standard allocations** - No memory pools
- **No async operations** - No CUDA streams
- **No memory hints** - Not using `cudaMemAdvise` or prefetching

## Missing CUDA Built-ins We Should Explore

### 1. **CUDA Streams** (HIGHEST IMPACT - 2-3x speedup potential)
**What we're missing:**
- All operations are synchronous
- No overlap between computation and memory transfers
- CPU blocks waiting for GPU operations

**What to implement:**
```rust
use cust::stream::{Stream, StreamFlags};

pub struct CudaHal<Hash: CudaHash + ?Sized> {
    pub max_threads: u32,
    hash: Option<Box<Hash>>,
    _context: Context,
    stream: Stream,  // ADD THIS
    _lock: ReentrantMutexGuard<'static, ()>,
}

// Use stream for async operations
fn copy_from_async(&self, stream: &Stream, src: &[u8], dst: DevicePointer<u8>) {
    unsafe {
        cudaMemcpyAsync(
            dst.as_raw_mut(),
            src.as_ptr(),
            src.len(),
            cudaMemcpyKind::cudaMemcpyHostToDevice,
            stream.as_raw()
        )
    }
}
```

**Benefits:**
- Overlap kernel execution with memory transfers
- 2-3x speedup for pipelined workloads
- Non-blocking operations

**Implementation effort:** Medium

---

### 2. **Async Memory Copies** (`cudaMemcpyAsync`)
**What we're missing:**
- All `copy_from()` calls are synchronous
- `as_host_vec()` blocks CPU
- No pipelining of transfers

**What to implement:**
```rust
// Replace synchronous copy_from with async version
impl RawBuffer {
    pub fn copy_from_async(&mut self, stream: &Stream, data: &[u8]) {
        unsafe {
            cudaMemcpyAsync(
                self.buf.as_device_ptr().as_raw_mut(),
                data.as_ptr(),
                data.len(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
                stream.as_raw()
            )
        }
    }

    // Async D2H transfer
    pub fn async_to_host(&self, stream: &Stream) -> Pin<Vec<u8>> {
        // Start async transfer, return future/pinned buffer
    }
}
```

**Benefits:**
- Overlap transfers with computation
- Non-blocking operations
- Better GPU utilization

**Implementation effort:** Medium

---

### 3. **Memory Pools** (`cudaMallocAsync` - CUDA 11.2+)
**What we're missing:**
- Standard `DeviceBuffer::uninitialized()` allocations
- No memory reuse
- Allocation overhead on every buffer creation

**What to implement:**
```rust
use cust::memory::MemoryPool;

static MEMORY_POOL: OnceLock<MemoryPool> = OnceLock::new();

impl RawBuffer {
    pub fn new_pooled(name: &'static str, size: usize) -> Self {
        let pool = MEMORY_POOL.get_or_init(|| {
            let device = Device::get_device(0).unwrap();
            MemoryPool::new(device).unwrap()
        });

        // Allocate from pool using cudaMallocAsync
        let ptr = unsafe {
            cudaMallocAsync(size, pool.as_raw(), stream.as_raw())
        };
        // Wrap in DeviceBuffer
    }
}
```

**Benefits:**
- 20-40% faster allocations
- Better memory reuse
- Reduced fragmentation
- Non-blocking allocations

**Implementation effort:** Medium-High

---

### 4. **Memory Advice Hints** (`cudaMemAdvise`)
**What we're missing:**
- No hints to CUDA about memory access patterns
- CUDA can't optimize memory placement

**What to implement:**
```rust
extern "C" {
    fn cudaMemAdvise(
        devPtr: *const ::std::os::raw::c_void,
        count: usize,
        advice: cudaMemAdvise,
        device: ::std::os::raw::c_int,
    ) -> cudaError_t;
}

#[repr(u32)]
enum cudaMemAdvise {
    SetReadMostly = 1,
    UnsetReadMostly = 2,
    SetPreferredLocation = 3,
    UnsetPreferredLocation = 4,
    SetAccessedBy = 5,
    UnsetAccessedBy = 6,
}

impl RawBuffer {
    // Mark input buffers as read-only (most accessed by GPU)
    pub fn advise_read_mostly(&self) {
        unsafe {
            cudaMemAdvise(
                self.buf.as_device_ptr().as_raw() as *const _,
                self.buf.len(),
                cudaMemAdvise::SetReadMostly,
                0 // Current device
            );
        }
    }

    // Mark output buffers as write-mostly
    pub fn advise_preferred_location_gpu(&self) {
        unsafe {
            cudaMemAdvise(
                self.buf.as_device_ptr().as_raw() as *const _,
                self.buf.len(),
                cudaMemAdvise::SetPreferredLocation,
                0
            );
        }
    }
}
```

**Benefits:**
- Better memory placement
- Optimized cache behavior
- 10-20% improvement for memory-bound operations

**Implementation effort:** Low

---

### 5. **Page-Locked (Pinned) Memory** for Host Buffers
**What we're missing:**
- Host memory in `cached_host` is pageable
- Slower transfers than pinned memory

**What to implement:**
```rust
use cust::memory::HostBuffer;

struct RawBuffer {
    // Use pinned memory for cached host copy
    cached_host: Option<HostBuffer<u8>>,  // Instead of Vec<u8>
}

impl RawBuffer {
    fn get_host_copy_pinned(&mut self) -> &[u8] {
        if self.cached_host.is_none() {
            // Allocate pinned memory
            self.cached_host = Some(
                HostBuffer::uninitialized(self.buf.len()).unwrap()
            );
            // Transfer to pinned memory
        }
        self.cached_host.as_ref().unwrap().as_slice()
    }
}
```

**Benefits:**
- 2-3x faster host-device transfers
- Can be used with async operations
- Better for frequent transfers

**Implementation effort:** Medium

---

### 6. **Read-Only Cache Hints** (L1/L2 cache optimization)
**What we're missing:**
- No cache hints for read-only buffers
- CUDA can't optimize cache behavior

**What to implement:**
```rust
// Use __ldg() intrinsic in kernels for read-only data
// Or use texture memory for read-only buffers
// Or set cache hints via cudaMemAdvise
```

**Benefits:**
- Better cache utilization
- Faster reads for input buffers
- 5-15% improvement for read-heavy workloads

**Implementation effort:** Low-Medium

---

### 7. **Unified Memory with Selective Prefetching**
**What we're missing:**
- We disabled unified memory prefetching earlier
- Could use it more selectively

**What to implement:**
```rust
// Only prefetch for buffers we know will be accessed
// Use cudaMemPrefetchAsync with proper device hints
impl RawBuffer {
    pub fn prefetch_to_gpu(&self, stream: &Stream) {
        unsafe {
            cudaMemPrefetchAsync(
                self.buf.as_device_ptr().as_raw() as *const _,
                self.buf.len(),
                0, // Device 0
                stream.as_raw()
            );
        }
    }
}
```

**Benefits:**
- Reduce page faults
- Better memory locality
- 10-20% improvement for unified memory workloads

**Implementation effort:** Low (but need to be selective)

---

## Recommended Implementation Order

### Phase 1: Quick Wins (Low Risk, Medium Impact)
1. **Memory Advice Hints** - Easy to add, immediate benefit
2. **Read-Only Cache Hints** - Low risk, good for input buffers

### Phase 2: High Impact (Medium Risk, High Impact)
3. **CUDA Streams** - Biggest performance gain, requires careful synchronization
4. **Async Memory Copies** - Works with streams, enables pipelining

### Phase 3: Scalability (Medium-High Risk, High Impact)
5. **Memory Pools** - Requires CUDA 11.2+, more complex
6. **Page-Locked Memory** - Good for frequently transferred buffers

### Phase 4: Advanced (Higher Risk, Variable Impact)
7. **Selective Unified Memory Prefetching** - Need to be careful with sppark compatibility

---

## Expected Performance Gains

- **Current baseline:** Synchronous operations, standard allocations
- **After Phase 1:** +10-20% improvement
- **After Phase 2:** +100-200% improvement (2-3x speedup)
- **After Phase 3:** Additional +20-40% for allocation-heavy workloads
- **After Phase 4:** Additional +10-20% for specific workloads

**Total potential:** 2-4x speedup for GB10

---

## Implementation Notes

1. **Streams require synchronization** - Need to ensure kernels complete before using results
2. **Memory pools need CUDA 11.2+** - Check runtime version
3. **Async operations need careful error handling** - Use `cudaStreamSynchronize()` when needed
4. **Memory advice is hints only** - CUDA may ignore them, but they help when used correctly
5. **Pinned memory is limited** - Don't pin everything, only frequently transferred buffers

---

## Next Steps

1. Profile GB10 to identify bottlenecks
2. Start with Phase 1 (memory advice hints)
3. Add CUDA streams for async operations
4. Implement async memory copies
5. Add memory pools if CUDA version supports it
6. Measure and iterate

