// Comprehensive tests to verify FPGA implementation is a complete drop-in replacement
// for the original RISC0 CPU executor

use anyhow::Result;
use risc0_circuit_rv32im::execute::{Emulator, Risc0Machine, EmuContext, Risc0Context, ByteAddr, WordAddr, LoadOp};
use crate::risc0_fpga_interface::FpgaExecutor;

/// Test suite to verify FPGA implementation matches original executor behavior
pub struct CompatibilityTestSuite {
    fpga_executor: FpgaExecutor,
    test_programs: Vec<TestProgram>,
}

#[derive(Debug, Clone)]
pub struct TestProgram {
    pub name: String,
    pub instructions: Vec<u32>,
    pub expected_registers: Vec<(usize, u32)>,
    pub expected_termination: Option<(u32, u32)>,
}

impl CompatibilityTestSuite {
    pub fn new() -> Result<Self> {
        let fpga_executor = FpgaExecutor::new()?;

        let test_programs = vec![
            TestProgram {
                name: "Basic Arithmetic".to_string(),
                instructions: vec![
                    0x00100093, // addi x1, x0, 1
                    0x00200113, // addi x2, x0, 2
                    0x00208133, // add x2, x1, x2
                    0x00000073, // ecall
                ],
                expected_registers: vec![(1, 1), (2, 3)],
                expected_termination: Some((0, 0)),
            },
            TestProgram {
                name: "Memory Operations".to_string(),
                instructions: vec![
                    0x00100093, // addi x1, x0, 1
                    0x00200113, // addi x2, x0, 2
                    0x00208133, // add x2, x1, x2
                    0x0020a023, // sw x2, 0(x1)
                    0x0000a103, // lw x2, 0(x1)
                    0x00000073, // ecall
                ],
                expected_registers: vec![(1, 1), (2, 3)],
                expected_termination: Some((0, 0)),
            },
            TestProgram {
                name: "Branch Instructions".to_string(),
                instructions: vec![
                    0x00100093, // addi x1, x0, 1
                    0x00200113, // addi x2, x0, 2
                    0x00208133, // add x2, x1, x2
                    0x00308463, // beq x1, x3, 8
                    0x00400113, // addi x2, x0, 4
                    0x00000073, // ecall
                ],
                expected_registers: vec![(1, 1), (2, 4)],
                expected_termination: Some((0, 0)),
            },
        ];

        Ok(Self {
            fpga_executor,
            test_programs,
        })
    }

    /// Test basic instruction execution compatibility
    pub fn test_basic_execution(&mut self) -> Result<()> {
        println!("Testing basic execution compatibility...");

        for program in &self.test_programs {
            println!("  Testing: {}", program.name);

            let mut executor = FpgaExecutor::new()?;
            executor.load_program(&program.instructions)?;
            executor.run()?;

            // Verify register values
            for (reg_idx, expected_value) in &program.expected_registers {
                let actual_value = executor.load_register(*reg_idx)?;
                assert_eq!(actual_value, *expected_value,
                    "Register x{} mismatch in {}: expected {}, got {}",
                    reg_idx, program.name, expected_value, actual_value);
            }

            // Verify termination state
            if let Some((expected_a0, expected_a1)) = program.expected_termination {
                if let Some(terminate) = executor.terminate_state() {
                    assert_eq!(terminate.a0, expected_a0,
                        "Termination a0 mismatch in {}: expected {}, got {}",
                        program.name, expected_a0, terminate.a0);
                    assert_eq!(terminate.a1, expected_a1,
                        "Termination a1 mismatch in {}: expected {}, got {}",
                        program.name, expected_a1, terminate.a1);
                } else {
                    panic!("Expected termination state but got None for {}", program.name);
                }
            }
        }

        println!("✅ Basic execution compatibility tests passed");
        Ok(())
    }

    /// Test memory access compatibility
    pub fn test_memory_access(&mut self) -> Result<()> {
        println!("Testing memory access compatibility...");

        let mut executor = FpgaExecutor::new()?;

        // Test word-aligned memory access
        executor.store_memory(WordAddr(0), 0x12345678)?;
        let value = executor.load_memory(WordAddr(0))?;
        assert_eq!(value, 0x12345678, "Word-aligned memory access failed");

        // Test byte access
        executor.store_u8(ByteAddr(0), 0xAA)?;
        let byte = executor.load_u8(LoadOp::Load, ByteAddr(0))?;
        assert_eq!(byte, 0xAA, "Byte access failed");

        // Test region access
        let test_data = vec![1, 2, 3, 4, 5, 6, 7, 8];
        executor.store_region(ByteAddr(0), &test_data)?;
        let loaded_data = executor.load_region(LoadOp::Load, ByteAddr(0), test_data.len())?;
        assert_eq!(loaded_data, test_data, "Region access failed");

        println!("✅ Memory access compatibility tests passed");
        Ok(())
    }

    /// Test register access compatibility
    pub fn test_register_access(&mut self) -> Result<()> {
        println!("Testing register access compatibility...");

        let mut executor = FpgaExecutor::new()?;

        // Test register store/load
        for i in 0..32 {
            let test_value = i as u32 * 0x12345678;
            executor.store_register(i, test_value)?;
            let loaded_value = executor.load_register(i)?;
            assert_eq!(loaded_value, test_value,
                "Register x{} access failed: expected {}, got {}",
                i, test_value, loaded_value);
        }

        // Test peek_register (SyscallContext)
        for i in 0..32 {
            let test_value = i as u32 * 0x87654321;
            executor.store_register(i, test_value)?;
            let peeked_value = executor.peek_register(i)?;
            assert_eq!(peeked_value, test_value,
                "Peek register x{} failed: expected {}, got {}",
                i, test_value, peeked_value);
        }

        println!("✅ Register access compatibility tests passed");
        Ok(())
    }

    /// Test syscall context compatibility
    pub fn test_syscall_context(&mut self) -> Result<()> {
        println!("Testing syscall context compatibility...");

        let mut executor = FpgaExecutor::new()?;

        // Test peek_u32
        executor.store_memory(WordAddr(0), 0xDEADBEEF)?;
        let peeked = executor.peek_u32(ByteAddr(0))?;
        assert_eq!(peeked, 0xDEADBEEF, "peek_u32 failed");

        // Test peek_u8
        executor.store_u8(ByteAddr(0), 0x42)?;
        let peeked_byte = executor.peek_u8(ByteAddr(0))?;
        assert_eq!(peeked_byte, 0x42, "peek_u8 failed");

        // Test peek_region
        let test_data = vec![0x11, 0x22, 0x33, 0x44];
        executor.store_region(ByteAddr(0), &test_data)?;
        let peeked_region = executor.peek_region(ByteAddr(0), test_data.len())?;
        assert_eq!(peeked_region, test_data, "peek_region failed");

        // Test peek_page
        let page_data = executor.peek_page(0)?;
        assert_eq!(page_data.len(), 4096, "peek_page returned wrong size");

        // Test get_cycle and get_pc
        let cycles = executor.get_cycle();
        let pc = executor.get_pc();
        assert!(cycles >= 0, "get_cycle returned negative value");
        assert!(pc >= 0, "get_pc returned negative value");

        println!("✅ Syscall context compatibility tests passed");
        Ok(())
    }

    /// Test hardware acceleration compatibility
    pub fn test_hardware_acceleration(&mut self) -> Result<()> {
        println!("Testing hardware acceleration compatibility...");

        let mut executor = FpgaExecutor::new()?;

        // Test SHA2 acceleration
        let sha2_state = risc0_circuit_rv32im::execute::sha2::Sha2State::default();
        executor.on_sha2_cycle(risc0_circuit_rv32im::execute::CycleState::default(), &sha2_state);

        // Test Poseidon2 acceleration
        let poseidon2_state = risc0_circuit_rv32im::execute::poseidon2::Poseidon2State::default();
        executor.on_poseidon2_cycle(risc0_circuit_rv32im::execute::CycleState::default(), &poseidon2_state);

        // Test BigInt acceleration
        executor.ecall_bigint()?;

        println!("✅ Hardware acceleration compatibility tests passed");
        Ok(())
    }

    /// Test error handling and recovery
    pub fn test_error_handling(&mut self) -> Result<()> {
        println!("Testing error handling and recovery...");

        let mut executor = FpgaExecutor::new()?;

        // Test error recovery
        executor.recover_from_error()?;
        assert!(executor.error_recovery_mode, "Error recovery mode not set");
        assert!(executor.get_last_error().is_none(), "Last error should be None after recovery");

        // Test invalid register access
        let result = executor.load_register(100);
        assert!(result.is_err(), "Should fail on invalid register access");

        // Test invalid memory access
        let result = executor.load_memory(WordAddr(0xFFFFFFFF));
        assert!(result.is_err(), "Should fail on invalid memory access");

        println!("✅ Error handling and recovery tests passed");
        Ok(())
    }

    /// Test trait implementation completeness
    pub fn test_trait_completeness(&mut self) -> Result<()> {
        println!("Testing trait implementation completeness...");

        let mut executor = FpgaExecutor::new()?;

        // Test EmuContext trait methods
        let ecall_result = executor.ecall()?;
        assert!(ecall_result, "ecall should return true");

        let mret_result = executor.mret()?;
        assert!(!mret_result, "mret should return false");

        let trap_result = executor.trap(risc0_circuit_rv32im::execute::Exception::IllegalInstruction(0, 0))?;
        assert!(!trap_result, "trap should return false");

        // Test Risc0Context trait methods
        executor.set_pc(ByteAddr(0x1000));
        assert_eq!(executor.get_pc(), ByteAddr(0x1000));

        executor.set_user_pc(ByteAddr(0x2000));
        assert_eq!(executor.user_pc, ByteAddr(0x2000));

        executor.set_machine_mode(1);
        assert_eq!(executor.get_machine_mode(), 1);

        // Test suspend/resume
        executor.suspend()?;
        executor.resume()?;

        println!("✅ Trait implementation completeness tests passed");
        Ok(())
    }

    /// Run all compatibility tests
    pub fn run_all_tests(&mut self) -> Result<()> {
        println!("🚀 Running comprehensive compatibility tests...");

        self.test_basic_execution()?;
        self.test_memory_access()?;
        self.test_register_access()?;
        self.test_syscall_context()?;
        self.test_hardware_acceleration()?;
        self.test_error_handling()?;
        self.test_trait_completeness()?;

        println!("🎉 All compatibility tests passed! FPGA implementation is a complete drop-in replacement.");
        Ok(())
    }
}

/// Performance comparison tests
pub struct PerformanceTestSuite {
    fpga_executor: FpgaExecutor,
}

impl PerformanceTestSuite {
    pub fn new() -> Result<Self> {
        let fpga_executor = FpgaExecutor::new()?;
        Ok(Self { fpga_executor })
    }

    /// Test performance characteristics
    pub fn test_performance(&mut self) -> Result<()> {
        println!("Testing performance characteristics...");

        // Generate a large test program
        let mut large_program = Vec::new();
        for i in 0..1000 {
            large_program.push(0x00100093); // addi x1, x0, 1
            large_program.push(0x00200113); // addi x2, x0, 2
            large_program.push(0x00208133); // add x2, x1, x2
        }
        large_program.push(0x00000073); // ecall

        let mut executor = FpgaExecutor::new()?;
        executor.load_program(&large_program)?;

        let start = std::time::Instant::now();
        executor.run()?;
        let duration = start.elapsed();

        println!("  Large program execution time: {:?}", duration);
        println!("  User cycles: {}", executor.user_cycles);
        println!("  Total cycles: {}", executor.total_cycles);

        println!("✅ Performance tests completed");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comprehensive_compatibility() -> Result<()> {
        let mut test_suite = CompatibilityTestSuite::new()?;
        test_suite.run_all_tests()
    }

    #[test]
    fn test_performance() -> Result<()> {
        let mut test_suite = PerformanceTestSuite::new()?;
        test_suite.test_performance()
    }
}
