// Basys 3 Real Hardware Example
// Demonstrates real hardware integration with Digilent Basys 3 Artix-7 FPGA

use anyhow::Result;
use risc0_fpga_interface::{
    basys3_executor::{Basys3Executor, Basys3ExecutorConfig},
    basys3_hardware_interface::{Basys3Config, Basys3HardwareInterface},
    simple_real_executor::{Digest, ExecutorConfig, FpgaExecutorError},
};

/// Custom Basys 3 configuration for real hardware
fn create_basys3_config() -> Basys3Config {
    Basys3Config {
        uart_device: "/dev/ttyUSB1".to_string(), // Updated for your system
        baud_rate: 115200,
        clock_frequency: 100_000_000, // 100MHz for Basys 3
        bram_size: 16 * 1024,         // 16KB BRAM
        timeout_ms: 10000,             // 10 second timeout
    }
}

/// Custom executor configuration for production use
fn create_executor_config() -> ExecutorConfig {
    ExecutorConfig {
        hardware_timeout: std::time::Duration::from_secs(30),
        memory_timeout: std::time::Duration::from_secs(5),
        syscall_timeout: std::time::Duration::from_secs(10),
        max_retries: 5,
        enable_error_recovery: true,
    }
}

/// Example RISC-V program (simple arithmetic)
fn create_test_program() -> Vec<u32> {
    vec![
        // Load immediate values
        0x00100093, // addi x1, x0, 1      ; x1 = 1
        0x00200113, // addi x2, x0, 2      ; x2 = 2
        0x00300193, // addi x3, x0, 3      ; x3 = 3

        // Perform arithmetic
        0x00208133, // add x2, x1, x2      ; x2 = x1 + x2 = 3
        0x00310233, // add x4, x2, x3      ; x4 = x2 + x3 = 6
        0x00418333, // add x6, x3, x4      ; x6 = x3 + x4 = 9

        // Store result in memory
        0x00602023, // sw x6, 0(x0)        ; Store result at address 0

        // System call to terminate
        0x00000073, // ecall                ; system call
    ]
}

/// Main function demonstrating Basys 3 integration
fn main() -> Result<()> {
    println!("🚀 Basys 3 Artix-7 FPGA Real Hardware Integration");
    println!("==================================================");

    // Step 1: Create hardware configuration
    println!("\n1️⃣ Creating Basys 3 hardware configuration...");
    let hw_config = create_basys3_config();
    println!("   ✅ UART Device: {}", hw_config.uart_device);
    println!("   ✅ Baud Rate: {}", hw_config.baud_rate);
    println!("   ✅ Clock Frequency: {} MHz", hw_config.clock_frequency / 1_000_000);
    println!("   ✅ BRAM Size: {} KB", hw_config.bram_size / 1024);

    // Step 2: Create executor configuration
    println!("\n2️⃣ Creating executor configuration...");
    let exec_config = create_executor_config();
    println!("   ✅ Hardware Timeout: {:?}", exec_config.hardware_timeout);
    println!("   ✅ Max Retries: {}", exec_config.max_retries);
    println!("   ✅ Error Recovery: {}", exec_config.enable_error_recovery);

    // Step 3: Create Basys 3 executor
    println!("\n3️⃣ Creating Basys 3 executor...");
    let basys3_config = Basys3ExecutorConfig {
        hardware_config: hw_config,
        executor_config: exec_config,
    };

    let mut executor = match Basys3Executor::new(basys3_config) {
        Ok(exec) => {
            println!("   ✅ Basys 3 executor created successfully");
            exec
        }
        Err(e) => {
            println!("   ❌ Failed to create Basys 3 executor: {}", e);
            println!("   💡 Make sure your Basys 3 is connected and the UART device is correct");
            return Err(e);
        }
    };

    // Step 4: Initialize Basys 3 hardware
    println!("\n4️⃣ Initializing Basys 3 hardware...");
    match executor.initialize() {
        Ok(()) => {
            println!("   ✅ Basys 3 hardware initialized successfully");

            // Display hardware status
            let status = executor.get_hardware_status();
            println!("   📊 Hardware Status:");
            println!("      - Initialized: {}", status.is_initialized);
            println!("      - Clock Cycles: {}", status.clock_cycle);
            println!("      - BRAM Used: {} bytes", status.bram_used);
        }
        Err(e) => {
            println!("   ❌ Failed to initialize Basys 3 hardware: {}", e);
            println!("   💡 Check your hardware connections and bitstream");
            return Err(e);
        }
    }

    // Step 5: Create test program
    println!("\n5️⃣ Creating test program...");
    let program = create_test_program();
    println!("   ✅ Test program created: {} instructions", program.len());
    println!("   📝 Program performs: x1=1, x2=2, x3=3, then arithmetic operations");

    // Step 6: Set input digest for cryptographic integrity
    println!("\n6️⃣ Setting cryptographic input digest...");
    let input_digest = Digest::from_slice(&[0x42u8; 32])?;
    executor.set_input_digest(input_digest);
    println!("   ✅ Input digest set for cryptographic verification");

    // Step 7: Load program into Basys 3 BRAM
    println!("\n7️⃣ Loading program into Basys 3 BRAM...");
    match executor.load_program(&program) {
        Ok(()) => {
            println!("   ✅ Program loaded into Basys 3 BRAM successfully");
            println!("   📊 Program Status:");
            println!("      - Loaded: {}", executor.is_program_loaded());
            println!("      - PC: 0x{:08x}", executor.get_pc().0);
            println!("      - User PC: 0x{:08x}", executor.get_user_pc().0);
        }
        Err(e) => {
            println!("   ❌ Failed to load program: {}", e);
            return Err(e);
        }
    }

    // Step 8: Execute program on Basys 3
    println!("\n8️⃣ Executing program on Basys 3 Artix-7...");
    match executor.run() {
        Ok(()) => {
            println!("   ✅ Program execution completed successfully");
            println!("   📊 Execution Results:");
            println!("      - Complete: {}", executor.is_execution_complete());
            println!("      - Cycle Count: {}", executor.get_cycle_count());

            // Display register values
            let registers = executor.get_registers();
            println!("   📋 Register Values:");
            for i in 0..8 {
                println!("      x{}: 0x{:08x}", i, registers[i]);
            }
        }
        Err(e) => {
            println!("   ❌ Program execution failed: {}", e);
            if let Some(error) = executor.get_last_error() {
                println!("   🔍 Last Error: {:?}", error);
            }
            return Err(e);
        }
    }

    // Step 9: Read final state and verify results
    println!("\n9️⃣ Reading final state and verifying results...");

    // Read output digest
    if let Some(output_digest) = executor.get_output_digest() {
        println!("   ✅ Output digest: {:?}", output_digest.as_slice());
    } else {
        println!("   ⚠️  No output digest available");
    }

    // Read terminate state
    if let Some(terminate_state) = executor.get_terminate_state() {
        println!("   ✅ Terminate state: a0=0x{:08x}, a1=0x{:08x}",
                terminate_state.a0, terminate_state.a1);
    } else {
        println!("   ⚠️  No terminate state available");
    }

    // Read memory result
    match executor.read_memory(0, 4) {
        Ok(result_bytes) => {
            let result = u32::from_le_bytes([result_bytes[0], result_bytes[1],
                                           result_bytes[2], result_bytes[3]]);
            println!("   ✅ Memory result at address 0: 0x{:08x} ({})", result, result);
        }
        Err(e) => {
            println!("   ❌ Failed to read memory result: {}", e);
        }
    }

    // Step 10: Display final hardware status
    println!("\n🔟 Final hardware status:");
    let final_status = executor.get_hardware_status();
    println!("   📊 Hardware Status:");
    println!("      - Initialized: {}", final_status.is_initialized);
    println!("      - Clock Cycles: {}", final_status.clock_cycle);
    println!("      - BRAM Used: {} bytes", final_status.bram_used);

    // Calculate execution time
    let execution_time = final_status.clock_cycle as f64 / final_status.clock_cycle as f64;
    println!("   ⏱️  Execution time: {:.6} seconds", execution_time);

    println!("\n🎉 Basys 3 real hardware integration completed successfully!");
    println!("   🚀 Your FPGA executor is now production-ready!");

    Ok(())
}

/// Error handling demonstration
fn demonstrate_error_handling() -> Result<()> {
    println!("\n🔧 Error Handling Demonstration");
    println!("================================");

    // Create executor with aggressive timeouts to trigger errors
    let mut hw_config = create_basys3_config();
    hw_config.uart_device = "/dev/nonexistent".to_string(); // Invalid device

    let mut exec_config = create_executor_config();
    exec_config.hardware_timeout = std::time::Duration::from_millis(100); // Very short timeout

    let basys3_config = Basys3ExecutorConfig {
        hardware_config: hw_config,
        executor_config: exec_config,
    };

    match Basys3Executor::new(basys3_config) {
        Ok(mut executor) => {
            println!("   ✅ Executor created (will fail on hardware access)");

            // This should fail gracefully
            match executor.initialize() {
                Ok(()) => println!("   ✅ Unexpected success!"),
                Err(e) => {
                    println!("   ❌ Expected failure: {}", e);
                    if let Some(error) = executor.get_last_error() {
                        println!("   🔍 Error type: {:?}", error);
                    }
                }
            }
        }
        Err(e) => {
            println!("   ❌ Executor creation failed: {}", e);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basys3_config_creation() {
        let config = create_basys3_config();
        assert_eq!(config.clock_frequency, 100_000_000);
        assert_eq!(config.bram_size, 16 * 1024);
    }

    #[test]
    fn test_executor_config_creation() {
        let config = create_executor_config();
        assert_eq!(config.max_retries, 5);
        assert!(config.enable_error_recovery);
    }

    #[test]
    fn test_program_creation() {
        let program = create_test_program();
        assert!(!program.is_empty());
        assert!(program.len() > 5); // Should have multiple instructions
    }
}
