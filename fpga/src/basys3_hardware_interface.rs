// Basys 3 Artix-7 FPGA Hardware Interface
// Provides real hardware communication for Digilent Basys 3 board

use anyhow::{bail, Result};
use std::time::Duration;
use std::io::{Read, Write};

/// Basys 3 Hardware Configuration
#[derive(Debug, Clone)]
pub struct Basys3Config {
    pub uart_device: String,
    pub baud_rate: u32,
    pub clock_frequency: u32, // 100MHz for Basys 3
    pub bram_size: usize,     // 16KB BRAM
    pub timeout_ms: u64,
}

impl Default for Basys3Config {
    fn default() -> Self {
        Self {
            uart_device: "/dev/ttyUSB0".to_string(),
            baud_rate: 115200,
            clock_frequency: 100_000_000, // 100MHz
            bram_size: 16 * 1024,         // 16KB
            timeout_ms: 5000,
        }
    }
}

/// UART Communication Interface
pub struct Basys3Uart {
    port: Option<Box<dyn serialport::SerialPort>>,
    #[allow(dead_code)]
    config: Basys3Config,
}

impl Basys3Uart {
    pub fn new(config: Basys3Config) -> Result<Self> {
        let port = serialport::new(&config.uart_device, config.baud_rate)
            .timeout(Duration::from_millis(config.timeout_ms))
            .open()
            .map_err(|e| anyhow::anyhow!("Failed to open UART port: {}", e))?;

        Ok(Self {
            port: Some(port),
            config,
        })
    }

    /// Send command to FPGA
    pub fn send_command(&mut self, command: &[u8]) -> Result<()> {
        if let Some(port) = &mut self.port {
            port.write_all(command)?;
            port.flush()?;
            Ok(())
        } else {
            bail!("UART port not available")
        }
    }

    /// Receive response from FPGA
    pub fn receive_response(&mut self, buffer: &mut [u8]) -> Result<usize> {
        if let Some(port) = &mut self.port {
            let bytes_read = port.read(buffer)?;
            Ok(bytes_read)
        } else {
            bail!("UART port not available")
        }
    }

    /// Send command and wait for response
    pub fn send_command_with_response(&mut self, command: &[u8], response: &mut [u8]) -> Result<usize> {
        self.send_command(command)?;
        self.receive_response(response)
    }
}

/// BRAM Memory Management for Basys 3
pub struct Basys3Bram {
    memory: Vec<u8>,
    #[allow(dead_code)]
    config: Basys3Config,
}

impl Basys3Bram {
    pub fn new(config: Basys3Config) -> Self {
        Self {
            memory: vec![0; config.bram_size],
            config,
        }
    }

    /// Write data to BRAM
    pub fn write(&mut self, address: u32, data: &[u8]) -> Result<()> {
        if address as usize + data.len() > self.memory.len() {
            bail!("BRAM write out of bounds: addr={}, len={}", address, data.len());
        }

        self.memory[address as usize..address as usize + data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Read data from BRAM
    pub fn read(&self, address: u32, length: usize) -> Result<Vec<u8>> {
        if address as usize + length > self.memory.len() {
            bail!("BRAM read out of bounds: addr={}, len={}", address, length);
        }

        Ok(self.memory[address as usize..address as usize + length].to_vec())
    }

    /// Get BRAM size
    pub fn size(&self) -> usize {
        self.memory.len()
    }
}

/// Basys 3 Clock Management
pub struct Basys3Clock {
    frequency: u32,
    current_cycle: u64,
}

impl Basys3Clock {
    pub fn new(frequency: u32) -> Self {
        Self {
            frequency,
            current_cycle: 0,
        }
    }

    /// Advance clock by one cycle
    pub fn tick(&mut self) {
        self.current_cycle += 1;
    }

    /// Get current cycle count
    pub fn get_cycle(&self) -> u64 {
        self.current_cycle
    }

    /// Get clock frequency
    pub fn get_frequency(&self) -> u32 {
        self.frequency
    }

    /// Calculate time from cycles
    pub fn cycles_to_time(&self, cycles: u64) -> Duration {
        let seconds = cycles as f64 / self.frequency as f64;
        Duration::from_secs_f64(seconds)
    }
}

/// Basys 3 Hardware Interface
pub struct Basys3HardwareInterface {
    uart: Basys3Uart,
    bram: Basys3Bram,
    clock: Basys3Clock,
    #[allow(dead_code)]
    config: Basys3Config,
    is_initialized: bool,
}

impl Basys3HardwareInterface {
    pub fn new(config: Basys3Config) -> Result<Self> {
        let uart = Basys3Uart::new(config.clone())?;
        let bram = Basys3Bram::new(config.clone());
        let clock = Basys3Clock::new(config.clock_frequency);

        Ok(Self {
            uart,
            bram,
            clock,
            config,
            is_initialized: false,
        })
    }

    /// Initialize Basys 3 hardware
    pub fn initialize(&mut self) -> Result<()> {
        println!("Initializing Basys 3 hardware...");

        // Send initialization command
        let init_cmd = b"INIT\n";
        self.uart.send_command(init_cmd)?;

        // Wait for acknowledgment
        let mut response = [0u8; 64];
        let bytes_read = self.uart.receive_response(&mut response)?;

        if bytes_read > 0 && &response[..bytes_read] == b"OK\n" {
            self.is_initialized = true;
            println!("✅ Basys 3 hardware initialized successfully");
            Ok(())
        } else {
            bail!("Failed to initialize Basys 3 hardware")
        }
    }

    /// Load program into BRAM
    pub fn load_program(&mut self, program: &[u32]) -> Result<()> {
        if !self.is_initialized {
            bail!("Hardware not initialized");
        }

        println!("Loading program into BRAM: {} instructions", program.len());

        // Convert program to bytes
        let program_bytes: Vec<u8> = program.iter()
            .flat_map(|&word| word.to_le_bytes().to_vec())
            .collect();

        // Write to BRAM
        self.bram.write(0, &program_bytes)?;

        // Send load command
        let load_cmd = format!("LOAD {}\n", program.len()).into_bytes();
        self.uart.send_command(&load_cmd)?;

        // Wait for acknowledgment
        let mut response = [0u8; 64];
        let bytes_read = self.uart.receive_response(&mut response)?;

        if bytes_read > 0 && &response[..bytes_read] == b"OK\n" {
            println!("✅ Program loaded into BRAM successfully");
            Ok(())
        } else {
            bail!("Failed to load program into BRAM")
        }
    }

    /// Execute program on Basys 3
    pub fn execute_program(&mut self) -> Result<()> {
        if !self.is_initialized {
            bail!("Hardware not initialized");
        }

        println!("Executing program on Basys 3...");

        // Send execute command
        let exec_cmd = b"EXEC\n";
        self.uart.send_command(exec_cmd)?;

        // Wait for completion
        let mut response = [0u8; 64];
        let bytes_read = self.uart.receive_response(&mut response)?;

        if bytes_read > 0 && &response[..bytes_read] == b"DONE\n" {
            println!("✅ Program execution completed");
            Ok(())
        } else {
            bail!("Program execution failed")
        }
    }

    /// Read register from Basys 3
    pub fn read_register(&mut self, reg_index: usize) -> Result<u32> {
        if !self.is_initialized {
            bail!("Hardware not initialized");
        }

        let cmd = format!("READ_REG {}\n", reg_index).into_bytes();
        let mut response = [0u8; 64];
        let bytes_read = self.uart.send_command_with_response(&cmd, &mut response)?;

        if bytes_read >= 8 {
            // Parse response: "REG xxxxxxxx\n"
            let response_str = std::str::from_utf8(&response[..bytes_read])?;
            if response_str.starts_with("REG ") {
                let value_str = &response_str[4..12]; // Skip "REG " and take 8 hex chars
                let value = u32::from_str_radix(value_str, 16)?;
                Ok(value)
            } else {
                bail!("Invalid register read response: {}", response_str)
            }
        } else {
            bail!("Invalid register read response")
        }
    }

    /// Write register to Basys 3
    pub fn write_register(&mut self, reg_index: usize, value: u32) -> Result<()> {
        if !self.is_initialized {
            bail!("Hardware not initialized");
        }

        let cmd = format!("WRITE_REG {} {:08x}\n", reg_index, value).into_bytes();
        let mut response = [0u8; 64];
        let bytes_read = self.uart.send_command_with_response(&cmd, &mut response)?;

        if bytes_read > 0 && &response[..bytes_read] == b"OK\n" {
            Ok(())
        } else {
            bail!("Failed to write register")
        }
    }

    /// Read memory from BRAM
    pub fn read_memory(&self, address: u32, length: usize) -> Result<Vec<u8>> {
        self.bram.read(address, length)
    }

    /// Write memory to BRAM
    pub fn write_memory(&mut self, address: u32, data: &[u8]) -> Result<()> {
        self.bram.write(address, data)
    }

    /// Get hardware status
    pub fn get_status(&self) -> Basys3Status {
        Basys3Status {
            is_initialized: self.is_initialized,
            clock_cycle: self.clock.get_cycle(),
            bram_used: self.bram.size(),
        }
    }

    /// Advance clock
    pub fn tick(&mut self) {
        self.clock.tick();
    }
}

/// Basys 3 Hardware Status
#[derive(Debug, Clone)]
pub struct Basys3Status {
    pub is_initialized: bool,
    pub clock_cycle: u64,
    pub bram_used: usize,
}

/// Basys 3 Hardware Error Types
#[derive(Debug, thiserror::Error)]
pub enum Basys3Error {
    #[error("UART communication error: {message}")]
    UartError { message: String },

    #[error("BRAM access error: {message}")]
    BramError { message: String },

    #[error("Hardware timeout: {operation}")]
    TimeoutError { operation: String },

    #[error("Hardware not initialized")]
    NotInitialized,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basys3_config() {
        let config = Basys3Config::default();
        assert_eq!(config.clock_frequency, 100_000_000);
        assert_eq!(config.bram_size, 16 * 1024);
    }

    #[test]
    fn test_basys3_bram() {
        let config = Basys3Config::default();
        let mut bram = Basys3Bram::new(config);

        // Test write and read
        let test_data = vec![0x12, 0x34, 0x56, 0x78];
        bram.write(0, &test_data).unwrap();

        let read_data = bram.read(0, 4).unwrap();
        assert_eq!(read_data, test_data);
    }

    #[test]
    fn test_basys3_clock() {
        let mut clock = Basys3Clock::new(100_000_000);
        assert_eq!(clock.get_cycle(), 0);

        clock.tick();
        assert_eq!(clock.get_cycle(), 1);

        let time = clock.cycles_to_time(100_000_000);
        assert_eq!(time.as_secs(), 1);
    }
}
