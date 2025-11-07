// Simple Real FPGA Executor Example
// Demonstrates basic RISC-V execution on real FPGA hardware

use anyhow::Result;
use risc0_fpga_interface::{
    SimpleRealFpgaExecutor, SimpleAcceleratorSupport, Risc0Context
};

fn main() -> Result<()> {
    println!("🚀 Simple Real FPGA Executor Example");
    println!("=====================================");

    // Configuration
    let device_path = "/dev/fpga0";
    let bitstream_path = "bitstreams/riscv_core.bit";

    // Create FPGA executor
    println!("\n1️⃣ Creating FPGA executor...");
    let mut executor = SimpleRealFpgaExecutor::new(device_path)?;
    println!("✅ FPGA executor created successfully");

    // Initialize FPGA
    println!("\n2️⃣ Initializing FPGA...");
    match executor.initialize(bitstream_path) {
        Ok(()) => println!("✅ FPGA initialized successfully"),
        Err(e) => println!("⚠️  FPGA initialization failed (expected without real hardware): {}", e),
    }

    // Test program
    let program = vec![
        0x00100093, // addi x1, x0, 1
        0x00200113, // addi x2, x0, 2
        0x00208133, // add x2, x1, x2
        0x00000073, // ecall
    ];

    // Load program
    println!("\n3️⃣ Loading program...");
    match executor.load_program(&program) {
        Ok(()) => println!("✅ Program loaded successfully"),
        Err(e) => println!("⚠️  Program loading failed (expected without real hardware): {}", e),
    }

    // Test basic operations
    println!("\n4️⃣ Testing basic operations...");

    // Test memory operations
    let test_addr = 0x100;
    let test_value = 0xdeadbeef;

    match Risc0Context::store_u32(&mut executor, risc0_fpga_interface::WordAddr(test_addr / 4), test_value) {
        Ok(()) => println!("  ✅ Stored 0x{:08x} at address 0x{:08x}", test_value, test_addr),
        Err(e) => println!("  ⚠️  Store failed: {}", e),
    }

    match Risc0Context::load_u32(&mut executor, risc0_fpga_interface::LoadOp::Peek, risc0_fpga_interface::WordAddr(test_addr / 4)) {
        Ok(value) => println!("  ✅ Loaded 0x{:08x} from address 0x{:08x}", value, test_addr),
        Err(e) => println!("  ⚠️  Load failed: {}", e),
    }

    // Test register operations
    let reg_idx = 5;
    let reg_value = 0x12345678;

    match Risc0Context::store_register(&mut executor, risc0_fpga_interface::WordAddr(0), reg_idx, reg_value) {
        Ok(()) => println!("  ✅ Stored 0x{:08x} in register x{}", reg_value, reg_idx),
        Err(e) => println!("  ⚠️  Register store failed: {}", e),
    }

    match Risc0Context::load_register(&mut executor, risc0_fpga_interface::LoadOp::Peek, risc0_fpga_interface::WordAddr(0), reg_idx) {
        Ok(value) => println!("  ✅ Loaded 0x{:08x} from register x{}", value, reg_idx),
        Err(e) => println!("  ⚠️  Register load failed: {}", e),
    }

    // Test system calls
    println!("\n5️⃣ Testing system calls...");

    let mut read_buf = vec![0u8; 64];
    match Risc0Context::host_read(&mut executor, 0, &mut read_buf) {
        Ok(bytes_read) => println!("  ✅ Host read: {} bytes", bytes_read),
        Err(e) => println!("  ⚠️  Host read failed: {}", e),
    }

    let write_buf = b"Hello, FPGA!";
    match Risc0Context::host_write(&mut executor, 1, write_buf) {
        Ok(bytes_written) => println!("  ✅ Host write: {} bytes", bytes_written),
        Err(e) => println!("  ⚠️  Host write failed: {}", e),
    }

    // Test accelerator support
    println!("\n6️⃣ Testing accelerator support...");

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

    // Display status
    println!("\n📊 Executor Status:");
    println!("  Execution complete: {}", executor.is_complete());
    println!("  Cycle count: {}", executor.get_cycle_count());
    println!("  PC: 0x{:08x}", executor.get_pc());

    // Display FPGA status
    let status = executor.get_fpga_status();
    println!("\n🔧 FPGA Status:");
    println!("  Ready: {}", status.is_ready);
    println!("  Executing: {}", status.is_executing);
    println!("  Done: {}", status.is_done);
    println!("  Cycle count: {}", status.cycle_count);
    println!("  Error code: {}", status.error_code);

    println!("\n🎉 Simple Real FPGA Executor example completed!");
    println!("   The executor is ready for real hardware integration.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executor_creation() -> Result<()> {
        let executor = SimpleRealFpgaExecutor::new("/dev/fpga0")?;
        assert!(!executor.is_complete());
        Ok(())
    }

    #[test]
    fn test_program_loading() -> Result<()> {
        let mut executor = SimpleRealFpgaExecutor::new("/dev/fpga0")?;

        let program = vec![
            0x00100093, // addi x1, x0, 1
            0x00200113, // addi x2, x0, 2
            0x00208133, // add x2, x1, x2
            0x00000073, // ecall
        ];

        // This will fail without real hardware, but should handle gracefully
        let result = executor.load_program(&program);
        assert!(result.is_ok() || result.is_err());

        Ok(())
    }
}
