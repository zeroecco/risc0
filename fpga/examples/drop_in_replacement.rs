// Example: Using FPGA Executor as Drop-in Replacement
// This demonstrates how to use the FPGA executor exactly like the original Rust emulator

use anyhow::Result;
use risc0_fpga_interface::FpgaExecutor;

fn main() -> Result<()> {
    println!("=== RISC0 FPGA Executor - Drop-in Replacement Example ===");

    // Create FPGA executor (replaces the original Rust emulator)
    let mut fpga_executor = FpgaExecutor::new()?;

    // Test program: addi x1, x0, 1; addi x2, x0, 2; add x2, x1, x2; ecall
    let program = vec![
        0x00100093, // addi x1, x0, 1
        0x00200113, // addi x2, x0, 2
        0x00208133, // add x2, x1, x2
        0x00000073, // ecall
    ];

    // Load program into FPGA
    fpga_executor.load_program(&program)?;

    println!("Executing program on FPGA...");

    // Execute using the same interface as original RISC0
    // This is exactly how you'd use the original Rust emulator
    let result = fpga_executor.run();

    match result {
        Ok(()) => {
            println!("✅ Execution completed successfully!");
            println!("Final PC: 0x{:08x}", fpga_executor.get_pc());
            println!("Register x1: {}", fpga_executor.registers[1]);
            println!("Register x2: {}", fpga_executor.registers[2]);

            if let Some(terminate) = fpga_executor.terminate_state() {
                println!("Termination: a0={}, a1={}", terminate.a0, terminate.a1);
            }
        }
        Err(e) => {
            println!("❌ Execution failed: {}", e);
        }
    }

    Ok(())
}

// Example: Performance comparison
#[allow(dead_code)]
fn benchmark_comparison() -> Result<()> {
    println!("\n=== Performance Comparison ===");

    // Create test program with more instructions
    let mut program = Vec::new();

    // Generate a larger program for testing
    for _i in 0..1000 {
        program.push(0x00100093); // addi x1, x0, 1
        program.push(0x00200113); // addi x2, x0, 2
        program.push(0x00208133); // add x2, x1, x2
    }
    program.push(0x00000073); // ecall

    // Test FPGA executor
    let start = std::time::Instant::now();
    let mut fpga_executor = FpgaExecutor::new()?;
    fpga_executor.load_program(&program)?;
    fpga_executor.run()?;
    let fpga_duration = start.elapsed();

    println!("FPGA Execution Time: {:?}", fpga_duration);
    println!("FPGA User Cycles: {}", fpga_executor.user_cycles);
    println!("FPGA Total Cycles: {}", fpga_executor.total_cycles);

    // In a real implementation, you'd compare this with the original Rust emulator
    // The FPGA should be significantly faster for large programs

    Ok(())
}

// Example: Integration with existing RISC0 workflow
#[allow(dead_code)]
fn integrate_with_risc0_workflow() -> Result<()> {
    println!("\n=== Integration with RISC0 Workflow ===");

    // This shows how the FPGA executor integrates with existing RISC0 code
    // The key is that it provides the same interface as the original executor

    let mut fpga_executor = FpgaExecutor::new()?;

    // You can use it anywhere the original executor is used
    // For example, in the RISC0 proving system:

    // 1. Load program
    let program = vec![
        0x00100093, // addi x1, x0, 1
        0x00200113, // addi x2, x0, 2
        0x00208133, // add x2, x1, x2
        0x00000073, // ecall
    ];
    fpga_executor.load_program(&program)?;

    // 2. Execute with same interface
    fpga_executor.run()?;

    // 3. Access results the same way
    let pc = fpga_executor.get_pc();
    let registers = &fpga_executor.registers;

    println!("Integration successful!");
    println!("PC: 0x{:08x}", pc);
    println!("Registers: {:?}", &registers[0..5]);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drop_in_replacement() -> Result<()> {
        let mut fpga_executor = FpgaExecutor::new()?;

        let program = vec![
            0x00100093, // addi x1, x0, 1
            0x00200113, // addi x2, x0, 2
            0x00208133, // add x2, x1, x2
            0x00000073, // ecall
        ];

        fpga_executor.load_program(&program)?;
        fpga_executor.run()?;

        // Verify results
        assert_eq!(fpga_executor.registers[1], 1);
        assert_eq!(fpga_executor.registers[2], 3);
        assert!(fpga_executor.terminate_state.is_some());

        Ok(())
    }
}
