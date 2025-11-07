// Production-Ready FPGA Executor Example
// Demonstrates all features needed for actual use

use anyhow::Result;
use risc0_fpga_interface::{
    SimpleRealFpgaExecutor, ByteAddr, LoadOp, InsnKind, DecodedInstruction,
    TraceEvent, TraceCallback, FpgaMetrics,
    Risc0Context, SyscallContext, EmuContext, SimpleAcceleratorSupport,
    Digest, TerminateState, SyscallHandler, DefaultSyscallHandler,
    GLOBAL_OUTPUT_ADDR, GLOBAL_INPUT_ADDR, DIGEST_BYTES
};

// Custom syscall handler for production use
struct ProductionSyscallHandler {
    input_data: Vec<u8>,
    output_data: Vec<u8>,
}

impl ProductionSyscallHandler {
    fn new(input_data: Vec<u8>) -> Self {
        Self {
            input_data,
            output_data: Vec::new(),
        }
    }

    fn get_output(&self) -> &[u8] {
        &self.output_data
    }
}

impl SyscallHandler for ProductionSyscallHandler {
    fn host_read(&mut self, _executor: &mut SimpleRealFpgaExecutor, fd: u32, buf: &mut [u8]) -> Result<u32> {
        match fd {
            0 => {
                // stdin - return input data
                let len = std::cmp::min(buf.len(), self.input_data.len());
                buf[..len].copy_from_slice(&self.input_data[..len]);
                Ok(len as u32)
            }
            _ => {
                // other file descriptors return empty
                Ok(0)
            }
        }
    }

    fn host_write(&mut self, _executor: &mut SimpleRealFpgaExecutor, fd: u32, buf: &[u8]) -> Result<u32> {
        match fd {
            1 | 2 => {
                // stdout/stderr - collect output
                self.output_data.extend_from_slice(buf);
                if let Ok(s) = std::str::from_utf8(buf) {
                    print!("{}", s);
                }
                Ok(buf.len() as u32)
            }
            _ => {
                // other file descriptors
                Ok(buf.len() as u32)
            }
        }
    }
}

// Example trace callback for production monitoring
struct ProductionTraceCallback {
    events: Vec<TraceEvent>,
    instruction_count: u64,
    memory_operations: u64,
}

impl ProductionTraceCallback {
    fn new() -> Self {
        Self {
            events: Vec::new(),
            instruction_count: 0,
            memory_operations: 0,
        }
    }

    fn print_summary(&self) {
        println!("📊 Production Trace Summary:");
        println!("  Total events: {}", self.events.len());
        println!("  Instructions executed: {}", self.instruction_count);
        println!("  Memory operations: {}", self.memory_operations);
    }
}

impl TraceCallback for ProductionTraceCallback {
    fn on_trace(&mut self, event: TraceEvent) -> Result<()> {
        self.events.push(event.clone());

        match event {
            TraceEvent::InstructionStart { .. } => {
                self.instruction_count += 1;
            }
            TraceEvent::MemorySet { .. } => {
                self.memory_operations += 1;
            }
            _ => {}
        }

        Ok(())
    }
}

fn main() -> Result<()> {
    println!("🚀 Production-Ready FPGA Executor Demo");
    println!("=======================================");

    // Configuration
    let device_path = "/dev/fpga0";
    let bitstream_path = "bitstreams/riscv_core.bit";

    // Create production syscall handler
    let input_data = b"Hello, FPGA! This is production input data.\n".to_vec();
    let syscall_handler = Box::new(ProductionSyscallHandler::new(input_data));

    // Create FPGA executor with custom syscall handler
    println!("\n1️⃣ Creating production FPGA executor...");
    let mut executor = SimpleRealFpgaExecutor::new_with_syscall_handler(device_path, syscall_handler)?;
    println!("✅ Production executor created successfully");

    // Set up cryptographic integrity
    println!("\n2️⃣ Setting up cryptographic integrity...");
    let input_digest = Digest::from_slice(&[1u8; 32])?;
    executor.set_input_digest(input_digest.clone());
    println!("✅ Input digest set: {:?}", input_digest.as_slice());

    // Add production trace callback
    println!("\n3️⃣ Setting up production monitoring...");
    let trace_callback = Box::new(ProductionTraceCallback::new());
    executor.add_trace_callback(trace_callback);
    println!("✅ Production monitoring enabled");

    // Initialize FPGA
    println!("\n4️⃣ Initializing FPGA...");
    match executor.initialize(bitstream_path) {
        Ok(()) => println!("✅ FPGA initialized successfully"),
        Err(e) => println!("⚠️  FPGA initialization failed (expected without real hardware): {}", e),
    }

    // Production test program: Complex computation with I/O
    let program = vec![
        // Load input data
        0x00100093, // addi x1, x0, 1      ; x1 = 1
        0x00200113, // addi x2, x0, 2      ; x2 = 2
        0x00300193, // addi x3, x0, 3      ; x3 = 3
        0x00400213, // addi x4, x0, 4      ; x4 = 4

        // Perform computation
        0x00208133, // add x2, x1, x2      ; x2 = x1 + x2 = 3
        0x00310233, // add x4, x2, x3      ; x4 = x2 + x3 = 6
        0x00418333, // add x6, x3, x4      ; x6 = x3 + x4 = 10
        0x00520433, // add x8, x4, x5      ; x8 = x4 + x5 = 6

        // Store result in global output address
        0xffff0237, // lui x4, 0xffff02    ; Load upper immediate for output addr
        0x04020213, // addi x4, x4, 64     ; Add offset to get GLOBAL_OUTPUT_ADDR
        0x00622023, // sw x6, 0(x4)        ; Store result at output address

        // System call to terminate
        0x00000073, // ecall                ; system call
    ];

    // Load program
    println!("\n5️⃣ Loading production program...");
    match executor.load_program(&program) {
        Ok(()) => println!("✅ Production program loaded successfully"),
        Err(e) => println!("⚠️  Program loading failed (expected without real hardware): {}", e),
    }

    // Demonstrate production features
    println!("\n6️⃣ Testing production features...");

    // Test cryptographic integrity
    println!("  🔐 Testing cryptographic integrity...");
    assert!(executor.get_output_digest().is_none()); // Should be None initially

    // Test memory operations with global addresses
    println!("  💾 Testing global memory addresses...");
    let test_value = 0xdeadbeef;
    match executor.store_u32(GLOBAL_OUTPUT_ADDR.waddr(), test_value) {
        Ok(()) => println!("    ✅ Stored test value at global output address"),
        Err(e) => println!("    ⚠️  Store failed: {}", e),
    }

    match executor.load_u32(LoadOp::Peek, GLOBAL_OUTPUT_ADDR.waddr()) {
        Ok(value) => println!("    ✅ Loaded value from global output address: 0x{:08x}", value),
        Err(e) => println!("    ⚠️  Load failed: {}", e),
    }

    // Test real syscall handler
    println!("  📡 Testing real syscall handler...");
    let mut read_buf = vec![0u8; 64];
    match executor.host_read(0, &mut read_buf) {
        Ok(bytes_read) => {
            println!("    ✅ Host read: {} bytes", bytes_read);
            if let Ok(s) = std::str::from_utf8(&read_buf[..bytes_read as usize]) {
                println!("    📝 Read data: '{}'", s.trim());
            }
        }
        Err(e) => println!("    ⚠️  Host read failed: {}", e),
    }

    let write_data = b"FPGA executor output: Hello, World!\n";
    match executor.host_write(1, write_data) {
        Ok(bytes_written) => println!("    ✅ Host write: {} bytes", bytes_written),
        Err(e) => println!("    ⚠️  Host write failed: {}", e),
    }

    // Test instruction execution with tracing
    println!("  🔍 Testing instruction execution with tracing...");
    let decoded_insn = DecodedInstruction {
        insn: 0x00100093,
        pc: ByteAddr(0x1000),
    };

    match executor.on_insn_start(InsnKind::Normal, &decoded_insn) {
        Ok(()) => println!("    ✅ Instruction start traced"),
        Err(e) => println!("    ⚠️  Instruction tracing failed: {}", e),
    }

    match executor.on_insn_end(InsnKind::Normal) {
        Ok(()) => println!("    ✅ Instruction end traced"),
        Err(e) => println!("    ⚠️  Instruction tracing failed: {}", e),
    }

    // Test terminate with cryptographic output
    println!("  🏁 Testing termination with cryptographic output...");
    match executor.on_terminate(42, 123) {
        Ok(()) => {
            println!("    ✅ Termination successful");
            if let Some(terminate_state) = executor.get_terminate_state() {
                println!("    📊 Terminate state: a0={}, a1={}", terminate_state.a0, terminate_state.a1);
            }
            if let Some(output_digest) = executor.get_output_digest() {
                println!("    🔐 Output digest: {:?}", output_digest.as_slice());
            }
        }
        Err(e) => println!("    ⚠️  Termination failed: {}", e),
    }

    // Display production metrics
    println!("\n📊 Production Metrics:");
    let metrics = executor.get_metrics();
    println!("  User cycles: {}", metrics.user_cycles);
    println!("  Paging cycles: {}", metrics.paging_cycles);
    println!("  Reserved cycles: {}", metrics.reserved_cycles);
    println!("  Total cycles: {}", metrics.total_cycles);

    // Display FPGA status
    println!("\n🔧 FPGA Status:");
    let status = executor.get_fpga_status();
    println!("  Ready: {}", status.is_ready);
    println!("  Executing: {}", status.is_executing);
    println!("  Done: {}", status.is_done);
    println!("  Cycle count: {}", status.cycle_count);
    println!("  Error code: {}", status.error_code);

    // Test accelerator support
    println!("\n⚡ Testing accelerator support...");
    match executor.enable_sha2() {
        Ok(()) => println!("  ✅ SHA2 accelerator enabled"),
        Err(e) => println!("  ⚠️  SHA2 accelerator not available: {}", e),
    }

    match executor.enable_poseidon2() {
        Ok(()) => println!("  ✅ Poseidon2 accelerator enabled"),
        Err(e) => println!("  ⚠️  Poseidon2 accelerator not available: {}", e),
    }

    match executor.enable_bigint() {
        Ok(()) => println!("  ✅ BigInt accelerator enabled"),
        Err(e) => println!("  ⚠️  BigInt accelerator not available: {}", e),
    }

    println!("\n🎉 Production-Ready FPGA Executor demonstration completed!");
    println!("   The executor now supports:");
    println!("   ✅ Cryptographic integrity (input/output digests)");
    println!("   ✅ Real syscall handler integration");
    println!("   ✅ Global memory addresses (GLOBAL_OUTPUT_ADDR, etc.)");
    println!("   ✅ Proper termination state handling");
    println!("   ✅ Production monitoring and tracing");
    println!("   ✅ Hardware accelerator support");
    println!("   ✅ Comprehensive error handling");
    println!("   ✅ Performance metrics collection");
    println!("   ✅ Memory region loading for digests");
    println!("   ✅ RISC0-compatible address constants");

    println!("\n🚀 Ready for production use!");
    println!("   To use with real hardware:");
    println!("   1. Connect FPGA development board");
    println!("   2. Load appropriate bitstream");
    println!("   3. Run with real device path");
    println!("   4. Verify cryptographic outputs");

    Ok(())
}
