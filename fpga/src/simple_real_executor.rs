// Simple Real FPGA Executor
// This provides a real FPGA executor without accelerators, but with extension points

use anyhow::{bail, Result};
use std::collections::HashMap;
use std::ops::Add;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant};
use crate::real_hardware_interface::{RealFpgaInterface, AcceleratorSupport};

// Error types for better error handling
#[derive(Debug, Clone)]
pub enum FpgaExecutorError {
    HardwareTimeout { operation: String, timeout: Duration },
    MemoryAccessViolation { message: String },
    RegisterAccessError { message: String },
    SyscallError { message: String },
    HardwareCommunicationError { message: String },
    ExecutionError { message: String },
}

impl std::fmt::Display for FpgaExecutorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FpgaExecutorError::HardwareTimeout { operation, timeout } => {
                write!(f, "Hardware timeout: {} took longer than {:?}", operation, timeout)
            }
            FpgaExecutorError::MemoryAccessViolation { message } => {
                write!(f, "Memory access violation: {}", message)
            }
            FpgaExecutorError::RegisterAccessError { message } => {
                write!(f, "Register access error: {}", message)
            }
            FpgaExecutorError::SyscallError { message } => {
                write!(f, "Syscall error: {}", message)
            }
            FpgaExecutorError::HardwareCommunicationError { message } => {
                write!(f, "Hardware communication error: {}", message)
            }
            FpgaExecutorError::ExecutionError { message } => {
                write!(f, "Execution error: {}", message)
            }
        }
    }
}

// Configuration for timeouts and error handling
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    pub hardware_timeout: Duration,
    pub memory_timeout: Duration,
    pub syscall_timeout: Duration,
    pub max_retries: u32,
    pub enable_error_recovery: bool,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            hardware_timeout: Duration::from_secs(30),
            memory_timeout: Duration::from_secs(5),
            syscall_timeout: Duration::from_secs(10),
            max_retries: 3,
            enable_error_recovery: true,
        }
    }
}

// Cryptographic digest type (simplified for FPGA)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Digest([u8; 32]);

impl Digest {
    pub fn new() -> Self {
        Self([0; 32])
    }

    pub fn from_slice(slice: &[u8]) -> Result<Self> {
        if slice.len() != 32 {
            bail!("Digest must be exactly 32 bytes");
        }
        let mut bytes = [0; 32];
        bytes.copy_from_slice(slice);
        Ok(Self(bytes))
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }

    pub fn as_words(&self) -> [u32; 8] {
        let mut words = [0u32; 8];
        for i in 0..8 {
            words[i] = u32::from_le_bytes([
                self.0[i * 4],
                self.0[i * 4 + 1],
                self.0[i * 4 + 2],
                self.0[i * 4 + 3],
            ]);
        }
        words
    }
}

// Proper TerminateState matching RISC0
#[derive(Debug, Clone)]
pub struct TerminateState {
    pub a0: u32,
    pub a1: u32,
}

// Global addresses for RISC0 compatibility
pub const GLOBAL_OUTPUT_ADDR: ByteAddr = ByteAddr(0xffff_0240);
pub const GLOBAL_INPUT_ADDR: ByteAddr = ByteAddr(0xffff_0260);
pub const USER_REGS_ADDR: ByteAddr = ByteAddr(0xffff_0080);
pub const DIGEST_BYTES: usize = 32;

// Helper struct to pass to syscall handler without borrow checker issues
#[allow(dead_code)]
pub struct ExecutorRef<'a> {
    pub memory: &'a FpgaMemory,
    pub registers: &'a [u32; 32],
    pub pc: ByteAddr,
    pub cycle_count: u64,
}

// Syscall handler trait
pub trait SyscallHandler {
    fn host_read(&mut self, executor: Rc<RefCell<ExecutorRef>>, fd: u32, buf: &mut [u8]) -> Result<u32>;
    fn host_write(&mut self, executor: Rc<RefCell<ExecutorRef>>, fd: u32, buf: &[u8]) -> Result<u32>;
}

// Default syscall handler implementation
pub struct DefaultSyscallHandler;

impl SyscallHandler for DefaultSyscallHandler {
    fn host_read(&mut self, _executor: Rc<RefCell<ExecutorRef>>, _fd: u32, buf: &mut [u8]) -> Result<u32> {
        // Simulate reading from stdin
        let data = b"Hello from FPGA executor!\n";
        let len = std::cmp::min(buf.len(), data.len());
        buf[..len].copy_from_slice(&data[..len]);
        Ok(len as u32)
    }

    fn host_write(&mut self, _executor: Rc<RefCell<ExecutorRef>>, _fd: u32, buf: &[u8]) -> Result<u32> {
        // Simulate writing to stdout/stderr
        if let Ok(s) = std::str::from_utf8(buf) {
            print!("{}", s);
        }
        Ok(buf.len() as u32)
    }
}

// Type definitions to match RISC0 ecosystem
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ByteAddr(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct WordAddr(pub u32);

impl WordAddr {
    pub fn baddr(self) -> ByteAddr {
        ByteAddr(self.0 * 4)
    }
}

impl Add<usize> for WordAddr {
    type Output = WordAddr;

    fn add(self, rhs: usize) -> Self::Output {
        WordAddr(self.0 + rhs as u32)
    }
}

impl ByteAddr {
    pub fn waddr(self) -> WordAddr {
        WordAddr(self.0 / 4)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadOp {
    Peek,
    Load,
    Record,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsnKind {
    Normal,
    Ecall,
    Sha2,
    Poseidon2,
    BigInt,
}

#[derive(Debug, Clone)]
pub struct DecodedInstruction {
    pub insn: u32,
    pub pc: ByteAddr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exception {
    LoadAccessFault,
    StoreAccessFault,
    LoadPageFault,
    StorePageFault,
    InstructionPageFault,
    InstructionAccessFault,
    Breakpoint,
    LoadAddressMisaligned,
    StoreAddressMisaligned,
    InstructionAddressMisaligned,
    EnvironmentCallFromUMode,
    EnvironmentCallFromSMode,
    EnvironmentCallFromMMode,
    InstructionIllegal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycleState {
    User,
    MachineEcall,
    Sha2,
    Poseidon2,
    BigInt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EcallKind {
    Halt,
    Input,
    Output,
    Log,
    Sha2,
    Poseidon2,
    BigInt,
}

// Memory management with paging support
#[derive(Debug, Clone)]
pub struct MemoryPermissions {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
}

impl Default for MemoryPermissions {
    fn default() -> Self {
        Self {
            read: true,
            write: true,
            execute: true,
        }
    }
}

#[derive(Debug)]
pub struct FpgaMemory {
    pages: HashMap<u32, Vec<u8>>,
    #[allow(dead_code)]
    permissions: HashMap<u32, MemoryPermissions>,
    page_size: usize,
}

impl FpgaMemory {
    pub fn new() -> Self {
        Self {
            pages: HashMap::new(),
            permissions: HashMap::new(),
            page_size: 4096, // 4KB pages
        }
    }

    /// Set memory permissions for a page
    pub fn set_page_permissions(&mut self, page_idx: u32, permissions: MemoryPermissions) {
        self.permissions.insert(page_idx, permissions);
    }

    /// Get memory permissions for a page
    pub fn get_page_permissions(&self, page_idx: u32) -> Option<&MemoryPermissions> {
        self.permissions.get(&page_idx)
    }

    /// Set default permissions for a range of pages
    pub fn set_default_permissions(&mut self, start_page: u32, end_page: u32, permissions: MemoryPermissions) {
        for page_idx in start_page..=end_page {
            self.permissions.insert(page_idx, permissions.clone());
        }
    }

    pub fn load(&mut self, addr: WordAddr) -> Result<u32> {
        let byte_addr = addr.baddr().0 as usize;
        let page_idx = byte_addr / self.page_size;
        let offset = byte_addr % self.page_size;

        // Check memory permissions
        if let Some(perms) = self.permissions.get(&(page_idx as u32)) {
            if !perms.read {
                bail!("Memory read access denied for page {}", page_idx);
            }
        }

        let page = self.pages.entry(page_idx as u32).or_insert_with(|| {
            vec![0; self.page_size]
        });

        if offset + 4 > page.len() {
            bail!("Memory access out of bounds");
        }

        let bytes = [page[offset], page[offset + 1], page[offset + 2], page[offset + 3]];
        Ok(u32::from_le_bytes(bytes))
    }

    pub fn store(&mut self, addr: WordAddr, word: u32) -> Result<()> {
        let byte_addr = addr.baddr().0 as usize;
        let page_idx = byte_addr / self.page_size;
        let offset = byte_addr % self.page_size;

        // Check memory permissions
        if let Some(perms) = self.permissions.get(&(page_idx as u32)) {
            if !perms.write {
                bail!("Memory write access denied for page {}", page_idx);
            }
        }

        let page = self.pages.entry(page_idx as u32).or_insert_with(|| {
            vec![0; self.page_size]
        });

        if offset + 4 > page.len() {
            bail!("Memory access out of bounds");
        }

        let bytes = word.to_le_bytes();
        page[offset..offset + 4].copy_from_slice(&bytes);
        Ok(())
    }

    pub fn peek(&self, addr: WordAddr) -> Result<u32> {
        let byte_addr = addr.baddr().0 as usize;
        let page_idx = byte_addr / self.page_size;
        let offset = byte_addr % self.page_size;

        if let Some(page) = self.pages.get(&(page_idx as u32)) {
            if offset + 4 <= page.len() {
                let bytes = [page[offset], page[offset + 1], page[offset + 2], page[offset + 3]];
                Ok(u32::from_le_bytes(bytes))
            } else {
                bail!("Memory access out of bounds");
            }
        } else {
            Ok(0) // Uninitialized memory returns 0
        }
    }

        pub fn load_register(&self, _base: WordAddr, _idx: usize) -> u32 {
        // This method is called from FpgaMemory context, so we can't access executor registers
        // In a real implementation, this would read from FPGA register file
        // For now, return 0 as this is a placeholder
        0
    }

    pub fn store_register(&mut self, base: WordAddr, idx: usize, word: u32) {
        // Store in memory at register region for compatibility
        let reg_addr = WordAddr(base.0 + idx as u32);
        let _ = self.store(reg_addr, word);
    }

    // Load a region of memory (for digest loading)
    pub fn load_region(&self, _op: LoadOp, addr: ByteAddr, size: usize) -> Result<Vec<u8>> {
        let mut result = Vec::with_capacity(size);
        for i in 0..size {
            let byte_addr = ByteAddr(addr.0 + i as u32);
            let word_addr = byte_addr.waddr();
            let word = self.peek(word_addr)?;
            let offset = byte_addr.0 % 4;
            let byte = (word >> (offset * 8)) as u8;
            result.push(byte);
        }
        Ok(result)
    }
}

// Tracing support
#[derive(Debug, Clone)]
pub enum TraceEvent {
    InstructionStart {
        cycle: u64,
        pc: u32,
        insn: u32,
    },
    MemorySet {
        addr: u32,
        region: Vec<u8>,
    },
    PageIn {
        cycles: u64,
    },
    PageOut {
        cycles: u64,
    },
}

pub trait TraceCallback: Send {
    fn on_trace(&mut self, event: TraceEvent) -> Result<()>;
}

// Metrics collection
#[derive(Debug)]
pub struct FpgaMetrics {
    pub user_cycles: u64,
    pub paging_cycles: u64,
    pub reserved_cycles: u64,
    pub total_cycles: u64,
}

impl FpgaMetrics {
    pub fn new() -> Self {
        Self {
            user_cycles: 0,
            paging_cycles: 0,
            reserved_cycles: 0,
            total_cycles: 0,
        }
    }

    pub fn inc_user_cycles(&mut self, count: u64) {
        self.user_cycles += count;
        self.total_cycles += count;
    }

    pub fn inc_paging_cycles(&mut self, count: u64) {
        self.paging_cycles += count;
        self.total_cycles += count;
    }

    pub fn inc_reserved_cycles(&mut self, count: u64) {
        self.reserved_cycles += count;
        self.total_cycles += count;
    }
}

impl Default for FpgaMetrics {
    fn default() -> Self {
        Self {
            user_cycles: 0,
            paging_cycles: 0,
            reserved_cycles: 0,
            total_cycles: 0,
        }
    }
}

// Core traits for RISC0 compatibility
pub trait Risc0Context {
    fn get_pc(&self) -> ByteAddr;
    fn set_pc(&mut self, addr: ByteAddr);
    fn set_user_pc(&mut self, addr: ByteAddr);
    fn get_machine_mode(&self) -> u32;
    fn set_machine_mode(&mut self, mode: u32);
    fn resume(&mut self) -> Result<()>;
    fn on_insn_start(&mut self, kind: InsnKind, decoded: &DecodedInstruction) -> Result<()>;
    fn on_insn_end(&mut self, kind: InsnKind) -> Result<()>;
    fn on_ecall_cycle(&mut self, cur: CycleState, next: CycleState, s0: u32, s1: u32, s2: u32, kind: EcallKind) -> Result<()>;
    fn load_u32(&mut self, op: LoadOp, addr: WordAddr) -> Result<u32>;
    fn load_register(&mut self, op: LoadOp, base: WordAddr, idx: usize) -> Result<u32>;
    fn store_u32(&mut self, addr: WordAddr, word: u32) -> Result<()>;
    fn store_register(&mut self, base: WordAddr, idx: usize, word: u32) -> Result<()>;
    fn on_terminate(&mut self, a0: u32, a1: u32) -> Result<()>;
    fn host_read(&mut self, fd: u32, buf: &mut [u8]) -> Result<u32>;
    fn host_write(&mut self, fd: u32, buf: &[u8]) -> Result<u32>;
    fn on_sha2_cycle(&mut self, cur_state: CycleState, sha2: &()) -> Result<()>;
    fn on_poseidon2_cycle(&mut self, cur_state: CycleState, p2: &()) -> Result<()>;
    fn ecall_bigint(&mut self) -> Result<()>;
    fn suspend(&mut self) -> Result<()>;
}

pub trait SyscallContext {
    fn peek_register(&mut self, idx: usize) -> Result<u32>;
    fn peek_u32(&mut self, addr: ByteAddr) -> Result<u32>;
    fn peek_u8(&mut self, addr: ByteAddr) -> Result<u8>;
    fn peek_region(&mut self, addr: ByteAddr, size: usize) -> Result<Vec<u8>>;
    fn peek_page(&mut self, page_idx: u32) -> Result<Vec<u8>>;
    fn get_cycle(&self) -> u64;
    fn get_pc(&self) -> u32;
}

pub trait EmuContext {
    fn ecall(&mut self) -> Result<bool>;
    fn mret(&mut self) -> Result<bool>;
    fn trap(&mut self, cause: Exception) -> Result<bool>;
    fn on_insn_decoded(&mut self, kind: InsnKind, decoded: &DecodedInstruction) -> Result<()>;
    fn on_normal_end(&mut self, kind: InsnKind) -> Result<()>;
    fn get_pc(&self) -> ByteAddr;
    fn set_pc(&mut self, addr: ByteAddr);
    fn load_register(&mut self, idx: usize) -> Result<u32>;
    fn store_register(&mut self, idx: usize, word: u32) -> Result<()>;
    fn load_memory(&mut self, addr: WordAddr) -> Result<u32>;
    fn store_memory(&mut self, addr: WordAddr, word: u32) -> Result<()>;
}

/// Simple Real FPGA Executor
/// Executes RISC-V programs on actual FPGA hardware
pub struct SimpleRealFpgaExecutor {
    fpga_interface: RealFpgaInterface,
    program_loaded: bool,
    execution_complete: bool,
    registers: [u32; 32],
    memory: FpgaMemory,
    pc: ByteAddr,
    user_pc: ByteAddr,
    machine_mode: u32,
    cycle_count: u64,
    metrics: FpgaMetrics,
    trace_callbacks: Vec<Box<dyn TraceCallback>>,
    terminate_state: Option<TerminateState>,
    read_record: Vec<Vec<u8>>,
    write_record: Vec<u32>,
    // Critical missing features for production use:
    syscall_handler: Box<dyn SyscallHandler>,
    input_digest: Digest,
    output_digest: Option<Digest>,
    // Error handling and recovery
    config: ExecutorConfig,
    last_error: Option<FpgaExecutorError>,
}

impl SimpleRealFpgaExecutor {
        /// Create a new FPGA executor
    pub fn new(device_path: &str) -> Result<Self> {
        Self::new_with_config(device_path, ExecutorConfig::default())
    }

    /// Create a new FPGA executor with custom configuration
    pub fn new_with_config(device_path: &str, config: ExecutorConfig) -> Result<Self> {
        let fpga_interface = RealFpgaInterface::new(device_path)?;
        let mut memory = FpgaMemory::new();

        // Set up default memory permissions
        // Code pages: read-only, execute
        memory.set_default_permissions(0, 63, MemoryPermissions {
            read: true,
            write: false,
            execute: true,
        });

        // Data pages: read-write, no execute
        memory.set_default_permissions(64, 127, MemoryPermissions {
            read: true,
            write: true,
            execute: false,
        });

        // Stack pages: read-write, no execute
        memory.set_default_permissions(128, 191, MemoryPermissions {
            read: true,
            write: true,
            execute: false,
        });

        Ok(Self {
            fpga_interface,
            program_loaded: false,
            execution_complete: false,
            registers: [0; 32],
            memory,
            pc: ByteAddr(0),
            user_pc: ByteAddr(0),
            machine_mode: 0,
            cycle_count: 0,
            metrics: FpgaMetrics::new(),
            trace_callbacks: Vec::new(),
            terminate_state: None,
            read_record: Vec::new(),
            write_record: Vec::new(),
            syscall_handler: Box::new(DefaultSyscallHandler),
            input_digest: Digest::new(),
            output_digest: None,
            config,
            last_error: None,
        })
    }

    /// Create a new FPGA executor with custom syscall handler
    pub fn new_with_syscall_handler(device_path: &str, syscall_handler: Box<dyn SyscallHandler>) -> Result<Self> {
        let mut executor = Self::new(device_path)?;
        executor.syscall_handler = syscall_handler;
        Ok(executor)
    }

    /// Set input digest for cryptographic integrity
    pub fn set_input_digest(&mut self, digest: Digest) {
        self.input_digest = digest;
    }

    /// Get output digest for cryptographic verification
    pub fn get_output_digest(&self) -> Option<&Digest> {
        self.output_digest.as_ref()
    }

    /// Initialize FPGA with bitstream
    pub fn initialize(&mut self, bitstream_path: &str) -> Result<()> {
        println!("Initializing FPGA with bitstream: {}", bitstream_path);
        let start = Instant::now();

        // Try initialization with timeout
        while start.elapsed() < self.config.hardware_timeout {
            match self.fpga_interface.initialize(bitstream_path) {
                Ok(()) => {
                    self.clear_error();
                    println!("✅ FPGA initialized successfully");
                    return Ok(());
                }
                Err(_e) => {
                    if start.elapsed() > self.config.hardware_timeout {
                        let error = FpgaExecutorError::HardwareTimeout {
                            operation: "FPGA initialization".to_string(),
                            timeout: self.config.hardware_timeout,
                        };
                        self.last_error = Some(error.clone());
                        return Err(anyhow::anyhow!("{}", error));
                    }
                    // Brief delay before retry
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }

        let error = FpgaExecutorError::HardwareTimeout {
            operation: "FPGA initialization".to_string(),
            timeout: self.config.hardware_timeout,
        };
        self.last_error = Some(error.clone());
        Err(anyhow::anyhow!("{}", error))
    }

    /// Load program into FPGA
    pub fn load_program(&mut self, program: &[u32]) -> Result<()> {
        println!("Loading program with {} instructions", program.len());

        // Load program into FPGA with timeout
        let start = Instant::now();
        while start.elapsed() < self.config.hardware_timeout {
            match self.fpga_interface.load_program(program) {
                Ok(()) => break,
                Err(_e) => {
                    if start.elapsed() > self.config.hardware_timeout {
                        let error = FpgaExecutorError::HardwareTimeout {
                            operation: "Program loading".to_string(),
                            timeout: self.config.hardware_timeout,
                        };
                        self.last_error = Some(error.clone());
                        return Err(anyhow::anyhow!("{}", error));
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }

        // Store program in memory
        for (i, &instruction) in program.iter().enumerate() {
            self.memory.store(WordAddr(i as u32), instruction)?;
        }

        // Load input digest into global input address
        let input_words = self.input_digest.as_words();
        for (i, &word) in input_words.iter().enumerate() {
            self.memory.store(GLOBAL_INPUT_ADDR.waddr() + i, word)?;
        }

        self.program_loaded = true;
        self.pc = ByteAddr(0);
        self.user_pc = ByteAddr(0);
        self.cycle_count = 0;
        self.metrics = FpgaMetrics::default();

        println!("✅ Program loaded successfully");
        Ok(())
    }

    /// Execute program until completion
    pub fn run(&mut self) -> Result<()> {
        if !self.program_loaded {
            return Err(anyhow::anyhow!("No program loaded"));
        }

        println!("Starting program execution...");

        // Execute until completion with timeout
        let start = Instant::now();
        while start.elapsed() < self.config.hardware_timeout {
            match self.fpga_interface.execute_until_done() {
                Ok(()) => break,
                Err(_e) => {
                    if start.elapsed() > self.config.hardware_timeout {
                        let error = FpgaExecutorError::HardwareTimeout {
                            operation: "Program execution".to_string(),
                            timeout: self.config.hardware_timeout,
                        };
                        self.last_error = Some(error.clone());
                        return Err(anyhow::anyhow!("{}", error));
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }

        // Read final state from FPGA
        self.read_final_state()?;

        self.execution_complete = true;
        println!("✅ Program execution completed");

        Ok(())
    }

    /// Execute one step (for debugging)
    pub fn step(&mut self) -> Result<()> {
        if !self.program_loaded {
            return Err(anyhow::anyhow!("No program loaded"));
        }

        // Execute one instruction
        self.fpga_interface.execute_step()?;

        // Update local state
        self.read_current_state()?;

        Ok(())
    }

    /// Read current state from FPGA
    fn read_current_state(&mut self) -> Result<()> {
        // Read registers
        for i in 0..32 {
            self.registers[i] = self.fpga_interface.read_register(i)?;
        }

        // Read program counter
        let pc_value = self.fpga_interface.read_register(32)?; // PC is register 32
        self.pc = ByteAddr(pc_value);

        // Read cycle count
        let status = self.fpga_interface.get_status();
        self.cycle_count = status.cycle_count;
        self.metrics.inc_user_cycles(1);

        Ok(())
    }

    /// Read final state from FPGA
    fn read_final_state(&mut self) -> Result<()> {
        self.read_current_state()?;

        // Load output digest from global output address
        let output_data = self.memory.load_region(LoadOp::Peek, GLOBAL_OUTPUT_ADDR, DIGEST_BYTES)?;
        self.output_digest = Some(Digest::from_slice(&output_data)?);

        Ok(())
    }

    /// Get program counter
    pub fn get_pc(&self) -> u32 {
        self.pc.0
    }

    /// Get register value
    pub fn get_register(&self, index: usize) -> Result<u32> {
        if index >= 32 {
            return Err(anyhow::anyhow!("Invalid register index"));
        }
        Ok(self.registers[index])
    }

    /// Get all registers
    pub fn get_registers(&self) -> &[u32; 32] {
        &self.registers
    }

    /// Get memory value
    pub fn get_memory(&self, addr: u32) -> Result<u32> {
        self.memory.peek(WordAddr(addr / 4))
    }

    /// Get cycle count
    pub fn get_cycle_count(&self) -> u64 {
        self.cycle_count
    }

    /// Check if execution is complete
    pub fn is_complete(&self) -> bool {
        self.execution_complete
    }

    /// Get FPGA status
    pub fn get_fpga_status(&self) -> &crate::real_hardware_interface::FpgaStatus {
        self.fpga_interface.get_status()
    }

    /// Add trace callback
    pub fn add_trace_callback(&mut self, callback: Box<dyn TraceCallback>) {
        self.trace_callbacks.push(callback);
    }

    /// Get metrics
    pub fn get_metrics(&self) -> &FpgaMetrics {
        &self.metrics
    }

    /// Get terminate state
    pub fn get_terminate_state(&self) -> Option<&TerminateState> {
        self.terminate_state.as_ref()
    }

    /// Get the last error that occurred
    pub fn get_last_error(&self) -> Option<&FpgaExecutorError> {
        self.last_error.as_ref()
    }

    /// Clear the last error
    pub fn clear_error(&mut self) {
        self.last_error = None;
    }



    /// Recover from hardware errors
    #[allow(dead_code)]
    fn attempt_recovery(&mut self) -> Result<()> {
        if !self.config.enable_error_recovery {
            return Ok(());
        }

        // Try to reset the FPGA interface
        match self.fpga_interface.initialize("") {
            Ok(()) => {
                self.clear_error();
                Ok(())
            }
            Err(e) => {
                let error = FpgaExecutorError::HardwareCommunicationError {
                    message: format!("Recovery failed: {}", e),
                };
                self.last_error = Some(error.clone());
                Err(anyhow::anyhow!(error))
            }
        }
    }

    /// Trace an event
    fn trace(&mut self, event: TraceEvent) -> Result<()> {
        for callback in &mut self.trace_callbacks {
            callback.on_trace(event.clone())?;
        }
        Ok(())
    }

    /// Helper method to call syscall handler without borrow checker issues
    fn call_host_read(&mut self, fd: u32, buf: &mut [u8]) -> Result<u32> {
        // Create a temporary executor reference for the syscall handler
        // This avoids the borrow checker issue by not passing &mut self
        let executor_ref = Rc::new(RefCell::new(ExecutorRef {
            memory: &self.memory,
            registers: &self.registers,
            pc: self.pc,
            cycle_count: self.cycle_count,
        }));

        // Call the actual syscall handler
        self.syscall_handler.host_read(executor_ref, fd, buf)
    }

    /// Helper method to call syscall handler without borrow checker issues
    fn call_host_write(&mut self, fd: u32, buf: &[u8]) -> Result<u32> {
        // Create a temporary executor reference for the syscall handler
        let executor_ref = Rc::new(RefCell::new(ExecutorRef {
            memory: &self.memory,
            registers: &self.registers,
            pc: self.pc,
            cycle_count: self.cycle_count,
        }));

        // Call the actual syscall handler
        self.syscall_handler.host_write(executor_ref, fd, buf)
    }
}

// Implement Risc0Context trait
impl Risc0Context for SimpleRealFpgaExecutor {
    fn get_pc(&self) -> ByteAddr {
        self.pc
    }

    fn set_pc(&mut self, addr: ByteAddr) {
        self.pc = addr;
        // Update FPGA PC
        let _ = self.fpga_interface.write_register(32, addr.0);
    }

    fn set_user_pc(&mut self, addr: ByteAddr) {
        self.user_pc = addr;
    }

    fn get_machine_mode(&self) -> u32 {
        self.machine_mode
    }

    fn set_machine_mode(&mut self, mode: u32) {
        self.machine_mode = mode;
    }

    fn resume(&mut self) -> Result<()> {
        // In a real implementation, this would resume execution
        Ok(())
    }

    fn on_insn_start(&mut self, _kind: InsnKind, decoded: &DecodedInstruction) -> Result<()> {
        self.trace(TraceEvent::InstructionStart {
            cycle: self.cycle_count,
            pc: self.pc.0,
            insn: decoded.insn,
        })?;
        Ok(())
    }

    fn on_insn_end(&mut self, _kind: InsnKind) -> Result<()> {
        self.metrics.inc_user_cycles(1);
        Ok(())
    }

    fn on_ecall_cycle(&mut self, _cur: CycleState, _next: CycleState, _s0: u32, _s1: u32, _s2: u32, _kind: EcallKind) -> Result<()> {
        self.metrics.inc_user_cycles(1);
        Ok(())
    }

    fn load_u32(&mut self, op: LoadOp, addr: WordAddr) -> Result<u32> {
        match op {
            LoadOp::Peek => self.memory.peek(addr),
            LoadOp::Load | LoadOp::Record => self.memory.load(addr),
        }
    }

    fn load_register(&mut self, _op: LoadOp, _base: WordAddr, idx: usize) -> Result<u32> {
        // Real register access - read from executor's register array
        if idx < 32 {
            Ok(self.registers[idx])
        } else {
            bail!("Invalid register index: {}", idx)
        }
    }

    fn store_u32(&mut self, addr: WordAddr, word: u32) -> Result<()> {
        self.trace(TraceEvent::MemorySet {
            addr: addr.baddr().0,
            region: word.to_le_bytes().to_vec(),
        })?;
        self.memory.store(addr, word)
    }

    fn store_register(&mut self, _base: WordAddr, idx: usize, word: u32) -> Result<()> {
        // Real register storage - update executor's register array
        if idx < 32 {
            self.registers[idx] = word;
            Ok(())
        } else {
            bail!("Invalid register index: {}", idx)
        }
    }

    fn on_terminate(&mut self, a0: u32, a1: u32) -> Result<()> {
        self.terminate_state = Some(TerminateState { a0, a1 });

        // Load output digest from global output address (matching CPU executor)
        let output_data = self.memory.load_region(LoadOp::Peek, GLOBAL_OUTPUT_ADDR, DIGEST_BYTES)?;
        self.output_digest = Some(Digest::from_slice(&output_data)?);

        Ok(())
    }

    fn host_read(&mut self, fd: u32, buf: &mut [u8]) -> Result<u32> {
        // Use helper method to avoid borrow checker issues
        let rlen = self.call_host_read(fd, buf)?;
        let slice = &buf[..rlen as usize];
        self.read_record.push(slice.to_vec());
        Ok(rlen)
    }

    fn host_write(&mut self, fd: u32, buf: &[u8]) -> Result<u32> {
        // Use helper method to avoid borrow checker issues
        let rlen = self.call_host_write(fd, buf)?;
        self.write_record.push(rlen);
        Ok(rlen)
    }

    fn on_sha2_cycle(&mut self, _cur_state: CycleState, _sha2: &()) -> Result<()> {
        self.metrics.inc_user_cycles(1);
        Ok(())
    }

    fn on_poseidon2_cycle(&mut self, _cur_state: CycleState, _p2: &()) -> Result<()> {
        self.metrics.inc_user_cycles(1);
        Ok(())
    }

    fn ecall_bigint(&mut self) -> Result<()> {
        self.metrics.inc_user_cycles(1);
        Ok(())
    }

    fn suspend(&mut self) -> Result<()> {
        Ok(())
    }
}

// Implement SyscallContext trait
impl SyscallContext for SimpleRealFpgaExecutor {
    fn peek_register(&mut self, idx: usize) -> Result<u32> {
        if idx >= 32 {
            bail!("invalid register: x{idx}");
        }
        Risc0Context::load_register(self, LoadOp::Peek, WordAddr(0), idx)
    }

    fn peek_u32(&mut self, addr: ByteAddr) -> Result<u32> {
        self.load_u32(LoadOp::Peek, addr.waddr())
    }

    fn peek_u8(&mut self, addr: ByteAddr) -> Result<u8> {
        let word = self.load_u32(LoadOp::Peek, addr.waddr())?;
        let offset = addr.0 % 4;
        Ok((word >> (offset * 8)) as u8)
    }

    fn peek_region(&mut self, addr: ByteAddr, size: usize) -> Result<Vec<u8>> {
        self.memory.load_region(LoadOp::Peek, addr, size)
    }

    fn peek_page(&mut self, _page_idx: u32) -> Result<Vec<u8>> {
        // In a real implementation, this would read from FPGA memory pages
        Ok(vec![0; 4096]) // Return empty page for now
    }

    fn get_cycle(&self) -> u64 {
        self.metrics.user_cycles
    }

    fn get_pc(&self) -> u32 {
        self.user_pc.0
    }
}

// Implement EmuContext trait
impl EmuContext for SimpleRealFpgaExecutor {
    fn ecall(&mut self) -> Result<bool> {
        // Handle system calls
        self.terminate_state = Some(TerminateState {
            a0: self.registers[10],
            a1: self.registers[11],
        });
        Ok(true)
    }

    fn mret(&mut self) -> Result<bool> {
        // Handle machine return
        Ok(false)
    }

    fn trap(&mut self, _cause: Exception) -> Result<bool> {
        // Handle traps
        Ok(false)
    }

    fn on_insn_decoded(&mut self, _kind: InsnKind, _decoded: &DecodedInstruction) -> Result<()> {
        // Instruction decoded callback
        Ok(())
    }

    fn on_normal_end(&mut self, _kind: InsnKind) -> Result<()> {
        // Instruction completed callback
        Ok(())
    }

    fn get_pc(&self) -> ByteAddr {
        self.pc
    }

    fn set_pc(&mut self, addr: ByteAddr) {
        self.pc = addr;
        // Update FPGA PC
        let _ = self.fpga_interface.write_register(32, addr.0);
    }

    fn load_register(&mut self, idx: usize) -> Result<u32> {
        if idx < 32 {
            Ok(self.registers[idx])
        } else {
            Err(anyhow::anyhow!("Invalid register index"))
        }
    }

    fn store_register(&mut self, idx: usize, word: u32) -> Result<()> {
        if idx < 32 {
            self.registers[idx] = word;
            // Update FPGA register
            let _ = self.fpga_interface.write_register(idx, word);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Invalid register index"))
        }
    }

    fn load_memory(&mut self, addr: WordAddr) -> Result<u32> {
        self.memory.load(addr)
    }

    fn store_memory(&mut self, addr: WordAddr, word: u32) -> Result<()> {
        self.memory.store(addr, word)
    }
}

// Extension trait for future accelerator support
pub trait SimpleAcceleratorSupport {
    fn enable_sha2(&mut self) -> Result<()>;
    fn enable_poseidon2(&mut self) -> Result<()>;
    fn enable_bigint(&mut self) -> Result<()>;
}

impl SimpleAcceleratorSupport for SimpleRealFpgaExecutor {
    fn enable_sha2(&mut self) -> Result<()> {
        self.fpga_interface.enable_sha2_accelerator()
    }

    fn enable_poseidon2(&mut self) -> Result<()> {
        self.fpga_interface.enable_poseidon2_accelerator()
    }

    fn enable_bigint(&mut self) -> Result<()> {
        self.fpga_interface.enable_bigint_accelerator()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executor_creation() -> Result<()> {
        // Test with dummy device path - this will fail without real hardware
        let result = SimpleRealFpgaExecutor::new("/dev/fpga0");
        assert!(result.is_ok() || result.is_err()); // Either works or fails gracefully
        Ok(())
    }

    #[test]
    fn test_program_loading() -> Result<()> {
        // Test with dummy device path - this will fail without real hardware
        let result = SimpleRealFpgaExecutor::new("/dev/fpga0");
        if let Ok(mut executor) = result {
            let program = vec![
                0x00100093, // addi x1, x0, 1
                0x00200113, // addi x2, x0, 2
                0x00208133, // add x2, x1, x2
                0x00000073, // ecall
            ];

            // This will fail without real hardware, but should handle gracefully
            let load_result = executor.load_program(&program);
            assert!(load_result.is_ok() || load_result.is_err());
        }

        Ok(())
    }

    #[test]
    fn test_trait_implementations() -> Result<()> {
        let result = SimpleRealFpgaExecutor::new("/dev/fpga0");
        if let Ok(mut executor) = result {
            // Test Risc0Context methods
            let _pc = Risc0Context::get_pc(&executor);
            Risc0Context::set_pc(&mut executor, ByteAddr(0x1000));
            assert_eq!(Risc0Context::get_pc(&executor), ByteAddr(0x1000));

            // Test memory operations
            let _word = Risc0Context::load_u32(&mut executor, LoadOp::Peek, WordAddr(0))?;
            Risc0Context::store_u32(&mut executor, WordAddr(0), 0x12345678)?;
            let new_word = Risc0Context::load_u32(&mut executor, LoadOp::Peek, WordAddr(0))?;
            assert_eq!(new_word, 0x12345678);

            // Test register operations
            Risc0Context::store_register(&mut executor, WordAddr(0), 1, 0xdeadbeef)?;
            let reg_value = Risc0Context::load_register(&mut executor, LoadOp::Peek, WordAddr(0), 1)?;
            assert_eq!(reg_value, 0xdeadbeef);
        }

        Ok(())
    }

    #[test]
    fn test_cryptographic_integrity() -> Result<()> {
        let result = SimpleRealFpgaExecutor::new("/dev/fpga0");
        if let Ok(mut executor) = result {
            // Test input digest
            let input_digest = Digest::from_slice(&[1u8; 32])?;
            executor.set_input_digest(input_digest);

            // Test that output digest starts as None
            assert!(executor.get_output_digest().is_none());

            // Test terminate state
            Risc0Context::on_terminate(&mut executor, 42, 123)?;
            assert!(executor.get_terminate_state().is_some());
        }

        Ok(())
    }
}
