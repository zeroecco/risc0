// Comprehensive FPGA Executor Example
// Demonstrates all RISC0-compatible features

use anyhow::Result;
use risc0_fpga_interface::{
    SimpleRealFpgaExecutor, ByteAddr, WordAddr, LoadOp, InsnKind, DecodedInstruction,
    Exception, CycleState, EcallKind, TraceEvent, TraceCallback, FpgaMetrics,
    Risc0Context, SyscallContext, EmuContext, SimpleAcceleratorSupport
};

// Example trace callback implementation
struct ExampleTraceCallback {
    events: Vec<TraceEvent>,
}

impl ExampleTraceCallback {
    fn new() -> Self {
        Self { events: Vec::new() }
    }

    fn print_summary(&self) {
        println!("📊 Trace Summary:");
        println!("  Total events: {}", self.events.len());

        let mut instruction_count = 0;
        let mut memory_count = 0;

        for event in &self.events {
            match event {
                TraceEvent::InstructionStart { .. } => instruction_count += 1,
                TraceEvent::MemorySet { .. } => memory_count += 1,
                _ => {}
            }
        }

        println!("  Instructions: {}", instruction_count);
        println!("  Memory operations: {}", memory_count);
    }
}

impl TraceCallback for ExampleTraceCallback {
    fn on_trace(&mut self, event: TraceEvent) -> Result<()> {
        self.events.push(event);
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("🚀 Comprehensive FPGA Executor Demo");
    println!("=====================================");

    // Configuration
    let device_path = "/dev/fpga0";
    let bitstream_path = "bitstreams/riscv_core.bit";

    // Create FPGA executor
    println!("\n1️⃣ Creating FPGA executor...");
    let mut executor = SimpleRealFpgaExecutor::new(device_path)?;
    println!("✅ Executor created successfully");

    // Add trace callback
    println!("\n2️⃣ Setting up tracing...");
    let trace_callback = Box::new(ExampleTraceCallback::new());
    executor.add_trace_callback(trace_callback);
    println!("✅ Tracing enabled");

    // Initialize FPGA
    println!("\n3️⃣ Initializing FPGA...");
    match executor.initialize(bitstream_path) {
        Ok(()) => println!("✅ FPGA initialized successfully"),
        Err(e) => println!("⚠️  FPGA initialization failed (expected without real hardware): {}", e),
    }

    // Test program: Complex arithmetic and memory operations
    let program = vec![
        0x00100093, // addi x1, x0, 1      ; x1 = 1
        0x00200113, // addi x2, x0, 2      ; x2 = 2
        0x00300193, // addi x3, x0, 3      ; x3 = 3
        0x00400213, // addi x4, x0, 4      ; x4 = 4
        0x00208133, // add x2, x1, x2      ; x2 = x1 + x2 = 3
        0x00310233, // add x4, x2, x3      ; x4 = x2 + x3 = 6
        0x00418333, // add x6, x3, x4      ; x6 = x3 + x4 = 10
        0x00520433, // add x8, x4, x5      ; x8 = x4 + x5 = 6
        0x00000073, // ecall                ; system call
    ];

    // Load program
    println!("\n4️⃣ Loading program...");
    match executor.load_program(&program) {
        Ok(()) => println!("✅ Program loaded successfully"),
        Err(e) => println!("⚠️  Program loading failed (expected without real hardware): {}", e),
    }

    // Demonstrate Risc0Context trait methods
    println!("\n5️⃣ Testing Risc0Context compatibility...");

    // Test PC operations
    let initial_pc = executor.get_pc();
    println!("  Initial PC: 0x{:08x}", initial_pc.0);

    executor.set_pc(ByteAddr(0x1000));
    println!("  Set PC to: 0x{:08x}", executor.get_pc().0);

    executor.set_user_pc(ByteAddr(0x2000));
    println!("  Set user PC to: 0x{:08x}", executor.get_user_pc().0);

    // Test memory operations
    println!("\n6️⃣ Testing memory operations...");

    // Store and load a word
    let test_addr = WordAddr(0x100);
    let test_value = 0xdeadbeef;

    match executor.store_u32(test_addr, test_value) {
        Ok(()) => println!("  ✅ Stored 0x{:08x} at address 0x{:08x}", test_value, test_addr.0 * 4),
        Err(e) => println!("  ⚠️  Store failed: {}", e),
    }

    match executor.load_u32(LoadOp::Peek, test_addr) {
        Ok(value) => println!("  ✅ Loaded 0x{:08x} from address 0x{:08x}", value, test_addr.0 * 4),
        Err(e) => println!("  ⚠️  Load failed: {}", e),
    }

    // Test register operations
    println!("\n7️⃣ Testing register operations...");

    let base_addr = WordAddr(0x200);
    let reg_idx = 5;
    let reg_value = 0x12345678;

    match executor.store_register(base_addr, reg_idx, reg_value) {
        Ok(()) => println!("  ✅ Stored 0x{:08x} in register x{}", reg_value, reg_idx),
        Err(e) => println!("  ⚠️  Register store failed: {}", e),
    }

    match executor.load_register(LoadOp::Peek, base_addr, reg_idx) {
        Ok(value) => println!("  ✅ Loaded 0x{:08x} from register x{}", value, reg_idx),
        Err(e) => println!("  ⚠️  Register load failed: {}", e),
    }

    // Demonstrate SyscallContext trait methods
    println!("\n8️⃣ Testing SyscallContext compatibility...");

    // Test peek operations
    match executor.peek_register(10) {
        Ok(value) => println!("  ✅ Peeked register x10: 0x{:08x}", value),
        Err(e) => println!("  ⚠️  Register peek failed: {}", e),
    }

    let peek_addr = ByteAddr(0x100);
    match executor.peek_u32(peek_addr) {
        Ok(value) => println!("  ✅ Peeked memory at 0x{:08x}: 0x{:08x}", peek_addr.0, value),
        Err(e) => println!("  ⚠️  Memory peek failed: {}", e),
    }

    match executor.peek_u8(peek_addr) {
        Ok(value) => println!("  ✅ Peeked byte at 0x{:08x}: 0x{:02x}", peek_addr.0, value),
        Err(e) => println!("  ⚠️  Byte peek failed: {}", e),
    }

    // Demonstrate EmuContext trait methods
    println!("\n9️⃣ Testing EmuContext compatibility...");

    // Test register operations
    match executor.load_register(1) {
        Ok(value) => println!("  ✅ Loaded register x1: 0x{:08x}", value),
        Err(e) => println!("  ⚠️  Register load failed: {}", e),
    }

    match executor.store_register(1, 0xabcdef12) {
        Ok(()) => println!("  ✅ Stored 0x{:08x} in register x1", 0xabcdef12),
        Err(e) => println!("  ⚠️  Register store failed: {}", e),
    }

    // Test memory operations
    let mem_addr = WordAddr(0x50);
    match executor.load_memory(mem_addr) {
        Ok(value) => println!("  ✅ Loaded memory at 0x{:08x}: 0x{:08x}", mem_addr.0 * 4, value),
        Err(e) => println!("  ⚠️  Memory load failed: {}", e),
    }

    match executor.store_memory(mem_addr, 0x87654321) {
        Ok(()) => println!("  ✅ Stored 0x{:08x} in memory at 0x{:08x}", 0x87654321, mem_addr.0 * 4),
        Err(e) => println!("  ⚠️  Memory store failed: {}", e),
    }

    // Test system calls
    println!("\n🔟 Testing system calls...");

    let mut read_buf = vec![0u8; 64];
    match executor.host_read(0, &mut read_buf) {
        Ok(bytes_read) => println!("  ✅ Host read: {} bytes", bytes_read),
        Err(e) => println!("  ⚠️  Host read failed: {}", e),
    }

    let write_buf = b"Hello, FPGA!";
    match executor.host_write(1, write_buf) {
        Ok(bytes_written) => println!("  ✅ Host write: {} bytes", bytes_written),
        Err(e) => println!("  ⚠️  Host write failed: {}", e),
    }

    // Test instruction callbacks
    println!("\n1️⃣1️⃣ Testing instruction callbacks...");

    let decoded_insn = DecodedInstruction {
        insn: 0x00100093,
        pc: ByteAddr(0x1000),
    };

    match executor.on_insn_start(InsnKind::Normal, &decoded_insn) {
        Ok(()) => println!("  ✅ Instruction start callback executed"),
        Err(e) => println!("  ⚠️  Instruction start callback failed: {}", e),
    }

    match executor.on_insn_end(InsnKind::Normal) {
        Ok(()) => println!("  ✅ Instruction end callback executed"),
        Err(e) => println!("  ⚠️  Instruction end callback failed: {}", e),
    }

    // Test accelerator support
    println!("\n1️⃣2️⃣ Testing accelerator support...");

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

    // Display metrics
    println!("\n📊 Performance Metrics:");
    let metrics = executor.get_metrics();
    println!("  User cycles: {}", metrics.user_cycles);
    println!("  Paging cycles: {}", metrics.paging_cycles);
    println!("  Reserved cycles: {}", metrics.reserved_cycles);
    println!("  Total cycles: {}", metrics.total_cycles);

    // Display FPGA status
    println!("\n🔧 FPGA Status:");
    let status = executor.get_fpga_status();
    println!("  Initialized: {}", status.is_initialized);
    println!("  Cycle count: {}", status.cycle_count);
    println!("  Execution complete: {}", executor.is_complete());

    println!("\n🎉 Comprehensive FPGA Executor demonstration completed!");
    println!("   The executor now supports:");
    println!("   ✅ Risc0Context trait compatibility");
    println!("   ✅ SyscallContext trait compatibility");
    println!("   ✅ EmuContext trait compatibility");
    println!("   ✅ Advanced memory management with paging");
    println!("   ✅ Comprehensive tracing support");
    println!("   ✅ Performance metrics collection");
    println!("   ✅ Hardware accelerator extension points");
    println!("   ✅ System call simulation");
    println!("   ✅ Instruction-level callbacks");

    Ok(())
}
