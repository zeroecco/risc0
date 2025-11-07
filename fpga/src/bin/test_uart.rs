use std::io::{Read, Write};
use std::time::Duration;
use serialport::{SerialPort, available_ports};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Testing UART Communication with Basys 3");
    println!("==========================================");

    // List available ports
    println!("Available serial ports:");
    for port in available_ports()? {
        println!("  {}", port.port_name);
    }

    // Open UART connection
    let mut port = serialport::new("/dev/ttyUSB1", 115200)
        .timeout(Duration::from_millis(1000))
        .open()?;

    println!("✅ Connected to /dev/ttyUSB1");

    // Send a simple test message
    let test_message = b"Hello FPGA!\n";
    port.write_all(test_message)?;
    println!("📤 Sent: {}", String::from_utf8_lossy(test_message));

    // Try to read response
    let mut buffer = [0; 128];
    match port.read(&mut buffer) {
        Ok(bytes_read) => {
            if bytes_read > 0 {
                let response = String::from_utf8_lossy(&buffer[..bytes_read]);
                println!("📥 Received: {}", response);
            } else {
                println!("📥 No response received (this is normal for echo mode)");
            }
        }
        Err(e) => {
            println!("❌ Read error: {}", e);
        }
    }

    // Test echo functionality
    println!("\n🔄 Testing echo functionality...");
    for i in 0..5 {
        let message = format!("Test {}\n", i);
        port.write_all(message.as_bytes())?;
        println!("📤 Sent: {}", message.trim());

        // Small delay
        std::thread::sleep(Duration::from_millis(100));

        // Try to read echo
        let mut buffer = [0; 128];
        match port.read(&mut buffer) {
            Ok(bytes_read) => {
                if bytes_read > 0 {
                    let response = String::from_utf8_lossy(&buffer[..bytes_read]);
                    println!("📥 Echo: {}", response.trim());
                } else {
                    println!("📥 No echo received");
                }
            }
            Err(e) => {
                println!("❌ Echo read error: {}", e);
            }
        }
    }

    println!("\n✅ UART test completed");
    Ok(())
}
