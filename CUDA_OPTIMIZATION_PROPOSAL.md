# CUDA Buffer Optimization Proposal for GB10

## Current Performance Issues

1. **Synchronous D2H Transfers**: `view()`, `view_mut()`, `get_at()`, `to_vec()` all use synchronous `as_host_vec()` which blocks CPU
2. **No Async Operations**: All memory operations are synchronous
3. **Inefficient Small Transfers**: `get_at()` transfers entire buffer for single element access
4. **No Memory Pooling**: Each allocation is independent, causing fragmentation

## CUDA 12.0/13.0 Features to Leverage

### 1. Async Memory Operations (`cudaMallocAsync` - CUDA 11.2+)

* **Benefit**: Non-blocking allocations, better memory pool management
* **Impact**: 10-30% reduction in allocation overhead
* **Implementation**: Use memory pools with async allocation

### 2. Memory Pools (`cudaMemPool_t` - CUDA 11.2+)

* **Benefit**: Reduced allocation overhead, better memory reuse
* **Impact**: 20-40% faster allocations for frequently allocated buffers
* **Implementation**: Per-thread or per-context memory pools

### 3. Async Memory Copies (`cudaMemcpyAsync`)

* **Benefit**: Overlap computation with memory transfers
* **Impact**: 2-3x speedup for operations that can overlap
* **Implementation**: Use CUDA streams for async copies

### 4. Host Memory Caching

* **Benefit**: Avoid repeated D2H transfers for frequently accessed buffers
* **Impact**: 5-10x speedup for `get_at()` and `view()` operations
* **Implementation**: Cache host-side copies with dirty tracking

## Proposed Optimizations

### Option 1: Host-Side Caching (Quick Win - High Impact)

**Strategy**: Cache host-side copies of buffers that are frequently accessed, only sync when needed.

```rust
struct RawBuffer {
    name: &'static str,
    buf: DeviceBuffer<u8>,
    cached_host: Option<Vec<u8>>,  // Cached host copy
    dirty: bool,                    // True if device has newer data
}

impl RawBuffer {
    fn get_host_copy(&mut self) -> &[u8] {
        if self.cached_host.is_none() || self.dirty {
            // Only transfer if cache is stale
            self.cached_host = Some(self.buf.as_host_vec().unwrap());
            self.dirty = false;
        }
        self.cached_host.as_ref().unwrap()
    }

    fn mark_dirty(&mut self) {
        self.dirty = true;
    }
}
```

**Benefits**:

* Eliminates redundant D2H transfers
* 5-10x faster for `view()` and `get_at()` on cached buffers
* Minimal code changes

**Trade-offs**:

* Uses extra host memory
* Need to track dirty state

### Option 2: Async Memory Operations (Medium Effort - Medium Impact)

**Strategy**: Use CUDA streams and async operations for non-blocking transfers.

```rust
// Add stream to CudaHal
pub struct CudaHal<Hash: CudaHash + ?Sized> {
    pub max_threads: u32,
    hash: Option<Box<Hash>>,
    _context: Context,
    stream: Stream,  // Add CUDA stream
    _lock: ReentrantMutexGuard<'static, ()>,
}

// Use async copies
fn copy_from_async(name: &'static str, slice: &[T], stream: &Stream) -> Self {
    // Use cudaMemcpyAsync instead of copy_from
}
```

**Benefits**:

* Overlap computation with transfers
* 2-3x speedup for operations that can be pipelined

**Trade-offs**:

* More complex synchronization
* Need to manage stream lifetimes

### Option 3: Memory Pooling (Medium Effort - High Impact for Allocations)

**Strategy**: Use CUDA memory pools to reduce allocation overhead.

```rust
// Create memory pool per context
static MEMORY_POOL: OnceLock<MemoryPool> = OnceLock::new();

impl RawBuffer {
    pub fn new_pooled(name: &'static str, size: usize) -> Self {
        let pool = MEMORY_POOL.get_or_init(|| {
            // Create CUDA memory pool
            // Use cudaMemPoolCreate, cudaMemPoolSetAttribute
        });
        // Allocate from pool using cudaMallocAsync
    }
}
```

**Benefits**:

* 20-40% faster allocations
* Better memory reuse
* Reduced fragmentation

**Trade-offs**:

* Requires CUDA 11.2+
* More complex memory management

### Option 4: Optimized Small Transfers (Quick Win - Low Impact)

**Strategy**: For `get_at()`, only transfer the specific element, not the entire buffer.

```rust
fn get_at(&self, idx: usize) -> T {
    let item_size = std::mem::size_of::<T>();
    let offset = (self.offset + idx) * item_size;

    // Only transfer the single element
    let mut result = vec![0u8; item_size];
    unsafe {
        cudaMemcpy(
            result.as_mut_ptr() as *mut _,
            self.as_device_ptr().offset(offset as isize).as_raw_mut(),
            item_size,
            cudaMemcpyDeviceToHost
        );
    }
    // Deserialize result
}
```

**Benefits**:

* Much faster for single element access
* Reduces unnecessary data transfer

**Trade-offs**:

* Still synchronous (but minimal overhead)
* More complex implementation

## Recommended Implementation Order

1. **Phase 1 (Quick Win)**: Option 4 - Optimized small transfers
   * Low risk, immediate benefit for `get_at()`
   * Estimated: 10-20% improvement for single-element access

2. **Phase 2 (High Impact)**: Option 1 - Host-side caching
   * Medium risk, high benefit for read-heavy workloads
   * Estimated: 5-10x improvement for cached buffer access

3. **Phase 3 (Scalability)**: Option 3 - Memory pooling
   * Medium risk, high benefit for allocation-heavy workloads
   * Estimated: 20-40% faster allocations

4. **Phase 4 (Advanced)**: Option 2 - Async operations
   * Higher risk, requires careful synchronization
   * Estimated: 2-3x speedup for pipelined operations

## Code Changes Required

### Minimal Changes (Option 1 + 4):

* Add caching fields to `RawBuffer`
* Modify `get_at()` to transfer only needed element
* Add dirty tracking

### Full Optimization (All Options):

* Add CUDA stream to `CudaHal`
* Implement memory pool
* Add async copy methods
* Refactor buffer access patterns

## Performance Expectations

* **Current**: ~100% CPU blocking on D2H transfers
* **After Phase 1+2**: ~10-20% CPU blocking, 5-10x faster cached access
* **After Phase 3**: Additional 20-40% faster allocations
* **After Phase 4**: 2-3x overall speedup for pipelined workloads

## Compatibility

* **CUDA 11.2+**: Required for memory pools and async allocation
* **CUDA 12.0+**: Better memory pool features
* **CUDA 13.0+**: Enhanced unified memory support (if we revisit unified memory)
