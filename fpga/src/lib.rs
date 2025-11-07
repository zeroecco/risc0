// RISC0 FPGA Interface Library
// Provides a drop-in replacement for the original RISC0 Rust emulator

pub mod simplified_fpga_interface;
pub mod real_hardware_interface;
pub mod simple_real_executor;
pub mod basys3_hardware_interface;
pub mod basys3_executor;

pub use simplified_fpga_interface::*;
pub use real_hardware_interface::*;

// Re-export core types for RISC0 compatibility
pub use simple_real_executor::{
    ByteAddr, WordAddr, LoadOp, InsnKind, DecodedInstruction, Exception,
    CycleState, EcallKind, MemoryPermissions, FpgaMemory, TraceEvent,
    TraceCallback, FpgaMetrics, Risc0Context, SyscallContext, EmuContext,
    SimpleRealFpgaExecutor, SimpleAcceleratorSupport,
    // New cryptographic and syscall types
    Digest, TerminateState, SyscallHandler, DefaultSyscallHandler,
    GLOBAL_OUTPUT_ADDR, GLOBAL_INPUT_ADDR, USER_REGS_ADDR, DIGEST_BYTES
};
