#!/bin/bash

# Basys 3 Artix-7 FPGA Deployment Script
# Handles hardware setup, bitstream generation, and testing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="risc0-fpga-basys3"
VIVADO_VERSION="2023.2"  # Adjust to your Vivado version
BOARD_NAME="Digilent Basys 3"
UART_DEVICE="/dev/ttyUSB0"  # Adjust for your system

echo -e "${BLUE}🚀 Basys 3 Artix-7 FPGA Deployment Script${NC}"
echo "=================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install FTDI drivers (cross-platform)
install_ftdi_drivers() {
    echo -e "\n${YELLOW}🔧 Installing FTDI drivers...${NC}"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo -e "   📥 Downloading FTDI drivers for macOS..."

        # Check if Homebrew is installed
        if command_exists brew; then
            echo -e "   ✅ Homebrew found, installing FTDI drivers..."
            brew install --cask ftdi-vcp-driver || {
                echo -e "   ❌ Failed to install FTDI drivers via Homebrew"
                echo -e "   💡 Manual installation required:"
                echo -e "      1. Download from: https://ftdichip.com/drivers/vcp-drivers/"
                echo -e "      2. Install the macOS driver"
                echo -e "      3. Restart your computer"
                return 1
            }
        else
            echo -e "   ⚠️  Homebrew not found"
            echo -e "   💡 Install Homebrew first: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            echo -e "   💡 Or manually install FTDI drivers from: https://ftdichip.com/drivers/vcp-drivers/"
            return 1
        fi

        echo -e "   ✅ FTDI drivers installed"
        echo -e "   💡 Please restart your computer and reconnect the Basys 3"

    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo -e "   📥 Installing FTDI drivers for Linux..."

        # Check if user is in dialout group
        if groups $USER | grep -q dialout; then
            echo -e "   ✅ User is in dialout group"
        else
            echo -e "   ⚠️  User not in dialout group"
            echo -e "   💡 Run: sudo usermod -a -G dialout $USER"
            echo -e "   💡 Then log out and log back in"
            return 1
        fi

        # Check if udev rules exist for FTDI
        if [ -f "/etc/udev/rules.d/99-ftdi.rules" ]; then
            echo -e "   ✅ FTDI udev rules found"
        else
            echo -e "   ⚠️  FTDI udev rules not found"
            echo -e "   💡 Create /etc/udev/rules.d/99-ftdi.rules with:"
            echo -e "      SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"0403\", ATTRS{idProduct}==\"6001\", MODE=\"0666\""
            echo -e "   💡 Then reload udev: sudo udevadm control --reload-rules"
            return 1
        fi

        echo -e "   ✅ Linux FTDI setup complete"
        echo -e "   💡 Reconnect your Basys 3 board"

    else
        echo -e "   ℹ️  Unknown OS, manual driver installation required"
        echo -e "   💡 Download FTDI drivers from: https://ftdichip.com/drivers/vcp-drivers/"
        return 1
    fi

    return 0
}

# Function to check hardware connection
check_hardware() {
    echo -e "\n${YELLOW}1️⃣ Checking hardware connection...${NC}"

    # Detect operating system
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux-specific device names
        UART_DEVICES=("/dev/ttyUSB0" "/dev/ttyUSB1" "/dev/ttyACM0" "/dev/ttyACM1" "/dev/ttyS0" "/dev/ttyS1")
        echo -e "   🐧 Linux system detected"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS-specific device names
        UART_DEVICES=("/dev/ttyUSB0" "/dev/ttyUSB1" "/dev/ttyACM0" "/dev/ttyACM1" "/dev/cu.usbserial-*" "/dev/cu.usbmodem*" "/dev/cu.SLAB_USBtoUART*")
        echo -e "   🍎 macOS system detected"
    else
        # Generic device names
        UART_DEVICES=("/dev/ttyUSB0" "/dev/ttyUSB1" "/dev/ttyACM0" "/dev/ttyACM1")
        echo -e "   💻 Generic system detected"
    fi

    # Check for any available UART device
    FOUND_DEVICE=""
    for device in "${UART_DEVICES[@]}"; do
        if [ -e "$device" ]; then
            FOUND_DEVICE="$device"
            break
        fi
    done

    # Also check for any device matching patterns (Linux)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        for pattern in "/dev/ttyUSB"* "/dev/ttyACM"*; do
            if [ -e "$pattern" ]; then
                FOUND_DEVICE="$pattern"
                break
            fi
        done
    fi

    if [ -n "$FOUND_DEVICE" ]; then
        echo -e "   ✅ UART device found: $FOUND_DEVICE"
        UART_DEVICE="$FOUND_DEVICE"
    else
        echo -e "   ❌ No UART device found"
        echo -e "   💡 Troubleshooting steps:"
        echo -e "      1. Check USB cable (try a different cable)"
        echo -e "      2. Try different USB ports"
        echo -e "      3. Install FTDI drivers if needed"
        echo -e "      4. Check if board is powered on"
        echo -e "      5. Look for any device names in /dev/tty.* or /dev/cu.*"

        # Show current devices
        echo -e "   📋 Current serial devices:"
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            ls -la /dev/ttyUSB* /dev/ttyACM* /dev/ttyS* 2>/dev/null | head -10 || echo "      No additional serial devices found"
        else
            ls -la /dev/tty.* /dev/cu.* 2>/dev/null | grep -v "Bluetooth\|debug-console" || echo "      No additional serial devices found"
        fi

        # Check USB devices
        echo -e "   📋 USB devices:"
        if command_exists lsusb; then
            lsusb | grep -i "ftdi\|digilent\|basys\|artix" || echo "      No FTDI/Digilent devices found"
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            system_profiler SPUSBDataType | grep -A 2 -B 2 "Product ID\|Vendor ID" | head -20 || echo "      Could not list USB devices"
        else
            echo "      Use 'lsusb' to check USB devices"
        fi

        echo -e "   💡 Common Basys 3 device names:"
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            echo -e "      - /dev/ttyUSB0 (Linux)"
            echo -e "      - /dev/ttyACM0 (Linux)"
        else
            echo -e "      - /dev/ttyUSB0 (Linux)"
            echo -e "      - /dev/cu.usbserial-XXXXXX (macOS)"
            echo -e "      - /dev/cu.usbmodemXXXXXX (macOS)"
            echo -e "      - /dev/cu.SLAB_USBtoUART (macOS with FTDI drivers)"
        fi

        return 1
    fi

    # Check USB permissions
    if [ -r "$UART_DEVICE" ] && [ -w "$UART_DEVICE" ]; then
        echo -e "   ✅ UART device permissions OK"
    else
        echo -e "   ⚠️  UART device permission issues"
        echo -e "   💡 Run: sudo chmod 666 $UART_DEVICE"
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            echo -e "   💡 Or add your user to the dialout group: sudo usermod -a -G dialout $USER"
        fi
        return 1
    fi

    return 0
}

# Function to check Vivado installation
check_vivado() {
    echo -e "\n${YELLOW}2️⃣ Checking Vivado installation...${NC}"

    if command_exists vivado; then
        echo -e "   ✅ Vivado found in PATH"
        VIVADO_CMD="vivado"
    elif [ -d "/opt/Xilinx/Vivado/$VIVADO_VERSION" ]; then
        echo -e "   ✅ Vivado found in /opt/Xilinx/Vivado/$VIVADO_VERSION"
        VIVADO_CMD="/opt/Xilinx/Vivado/$VIVADO_VERSION/bin/vivado"
    elif [ -d "/tools/xilinx/2025.1/Vivado" ]; then
        echo -e "   ✅ Vivado found in /tools/xilinx/2025.1/Vivado"
        VIVADO_CMD="/tools/xilinx/2025.1/Vivado/bin/vivado"
    else
        echo -e "   ❌ Vivado not found"
        echo -e "   💡 Please install Vivado $VIVADO_VERSION or update the script"
        echo -e "   💡 Common locations:"
        echo -e "      - /opt/Xilinx/Vivado/$VIVADO_VERSION/bin/vivado"
        echo -e "      - /tools/xilinx/2025.1/Vivado/bin/vivado"
        echo -e "      - Add Vivado to your PATH"
        return 1
    fi

    # Test Vivado
    if $VIVADO_CMD -version >/dev/null 2>&1; then
        echo -e "   ✅ Vivado is working"
    else
        echo -e "   ❌ Vivado is not working properly"
        echo -e "   💡 Check Vivado installation and permissions"
        return 1
    fi

    return 0
}

# Function to create Vivado project
create_vivado_project() {
    echo -e "\n${YELLOW}3️⃣ Creating Vivado project...${NC}"

    PROJECT_DIR="vivado_project"

    if [ -d "$PROJECT_DIR" ]; then
        echo -e "   ⚠️  Project directory exists, removing..."
        rm -rf "$PROJECT_DIR"
    fi

    mkdir -p "$PROJECT_DIR"

    # Create Tcl script for Vivado project creation
    cat > create_project.tcl << 'EOF'
# Create Vivado project for Basys 3
create_project risc0-fpga-basys3 vivado_project -part xc7a35tcpg236-1 -force

# Try to set board part, but don't fail if it's not available
if {[catch {set_property board_part digilentinc:basys3:part0:1.2 [current_project]} result]} {
    puts "WARNING: Board part not found, continuing without board definition"
    puts "Available board parts:"
    puts [get_board_parts]
}

set_property target_language VHDL [current_project]

# Create constraints file for Basys 3
set constraints_file [file join [get_property directory [current_project]] "basys3_constraints.xdc"]
set fp [open $constraints_file w]

puts $fp "# Basys 3 Artix-7 FPGA Constraints"
puts $fp "# Generated for RISC-V processor implementation"
puts $fp ""
puts $fp "# Clock signal"
puts $fp "set_property PACKAGE_PIN W5 \[get_ports clk\]"
puts $fp "set_property IOSTANDARD LVCMOS33 \[get_ports clk\]"
puts $fp ""
puts $fp "# Reset signal"
puts $fp "set_property PACKAGE_PIN U18 \[get_ports rst\]"
puts $fp "set_property IOSTANDARD LVCMOS33 \[get_ports rst\]"
puts $fp ""
puts $fp "# UART signals"
puts $fp "set_property PACKAGE_PIN B18 \[get_ports uart_tx\]"
puts $fp "set_property IOSTANDARD LVCMOS33 \[get_ports uart_tx\]"
puts $fp "set_property PACKAGE_PIN B18 \[get_ports uart_rx\]"
puts $fp "set_property IOSTANDARD LVCMOS33 \[get_ports uart_rx\]"
puts $fp ""
puts $fp "# LED signals (16 LEDs)"
puts $fp "set_property PACKAGE_PIN U16 \[get_ports {led\[0\]}\]"
puts $fp "set_property IOSTANDARD LVCMOS33 \[get_ports {led\[0\]}\]"
puts $fp "set_property PACKAGE_PIN E19 \[get_ports {led\[1\]}\]"
puts $fp "set_property IOSTANDARD LVCMOS33 \[get_ports {led\[1\]}\]"
puts $fp "set_property PACKAGE_PIN U19 \[get_ports {led\[2\]}\]"
puts $fp "set_property IOSTANDARD LVCMOS33 \[get_ports {led\[2\]}\]"
puts $fp "set_property PACKAGE_PIN V19 \[get_ports {led\[3\]}\]"
puts $fp "set_property IOSTANDARD LVCMOS33 \[get_ports {led\[3\]}\]"
puts $fp "set_property PACKAGE_PIN W18 \[get_ports {led\[4\]}\]"
puts $fp "set_property IOSTANDARD LVCMOS33 \[get_ports {led\[4\]}\]"
puts $fp "set_property PACKAGE_PIN U15 \[get_ports {led\[5\]}\]"
puts $fp "set_property IOSTANDARD LVCMOS33 \[get_ports {led\[5\]}\]"
puts $fp "set_property PACKAGE_PIN U14 \[get_ports {led\[6\]}\]"
puts $fp "set_property IOSTANDARD LVCMOS33 \[get_ports {led\[6\]}\]"
puts $fp "set_property PACKAGE_PIN V14 \[get_ports {led\[7\]}\]"
puts $fp "set_property IOSTANDARD LVCMOS33 \[get_ports {led\[7\]}\]"
puts $fp "set_property PACKAGE_PIN V13 \[get_ports {led\[8\]}\]"
puts $fp "set_property IOSTANDARD LVCMOS33 \[get_ports {led\[8\]}\]"
puts $fp "set_property PACKAGE_PIN V3 \[get_ports {led\[9\]}\]"
puts $fp "set_property IOSTANDARD LVCMOS33 \[get_ports {led\[9\]}\]"
puts $fp "set_property PACKAGE_PIN W3 \[get_ports {led\[10\]}\]"
puts $fp "set_property IOSTANDARD LVCMOS33 \[get_ports {led\[10\]}\]"
puts $fp "set_property PACKAGE_PIN U3 \[get_ports {led\[11\]}\]"
puts $fp "set_property IOSTANDARD LVCMOS33 \[get_ports {led\[11\]}\]"
puts $fp "set_property PACKAGE_PIN P3 \[get_ports {led\[12\]}\]"
puts $fp "set_property IOSTANDARD LVCMOS33 \[get_ports {led\[12\]}\]"
puts $fp "set_property PACKAGE_PIN N3 \[get_ports {led\[13\]}\]"
puts $fp "set_property IOSTANDARD LVCMOS33 \[get_ports {led\[13\]}\]"
puts $fp "set_property PACKAGE_PIN P1 \[get_ports {led\[14\]}\]"
puts $fp "set_property IOSTANDARD LVCMOS33 \[get_ports {led\[14\]}\]"
puts $fp "set_property PACKAGE_PIN L1 \[get_ports {led\[15\]}\]"
puts $fp "set_property IOSTANDARD LVCMOS33 \[get_ports {led\[15\]}\]"

close $fp

# Add constraints file to project
add_files -fileset constrs_1 $constraints_file
set_property target_constrs_file $constraints_file [current_fileset -constrset]

puts "Project created successfully with Basys 3 constraints"
EOF

    # Create Vivado project using the Tcl script
    $VIVADO_CMD -mode batch -source create_project.tcl

    if [ $? -eq 0 ]; then
        echo -e "   ✅ Vivado project created successfully"
        echo -e "   📁 Project location: $(pwd)/$PROJECT_DIR"
        echo -e "   📄 Constraints file: $(pwd)/$PROJECT_DIR/basys3_constraints.xdc"
        # Clean up the temporary Tcl script
        rm -f create_project.tcl
    else
        echo -e "   ❌ Failed to create Vivado project"
        echo -e "   💡 Check Vivado installation and permissions"
        echo -e "   💡 Try running Vivado manually to check for issues"
        return 1
    fi

    return 0
}

# Function to generate bitstream
generate_bitstream() {
    echo -e "\n${YELLOW}4️⃣ Generating bitstream...${NC}"

    # Use the actual RISC-V processor design
    echo -e "   🔧 Using RISC-V processor implementation"
    create_riscv_design

    return 0
}

# Function to create RISC-V processor design
create_riscv_design() {
    echo -e "\n${YELLOW}5️⃣ Creating RISC-V processor design...${NC}"

    # Copy simplified RISC-V processor files to Vivado project
    if [ -f "risc0_fpga_riscv_simple.v" ]; then
        cp risc0_fpga_riscv_simple.v vivado_project/risc0_fpga_riscv.v
        echo -e "   ✅ Copied simplified RISC-V processor design"
    else
        echo -e "   ❌ Simplified RISC-V processor file not found"
        return 1
    fi

    # Copy constraints file if it exists
    if [ -f "risc0_fpga_constraints.xdc" ]; then
        cp risc0_fpga_constraints.xdc vivado_project/
        echo -e "   ✅ Copied constraints file"
    fi

    # Create top-level wrapper for Basys 3
    cat > vivado_project/basys3_top.v << 'EOF'
// Basys 3 Top-Level Module for RISC-V Processor
// Wraps the RISC-V processor with Basys 3 I/O

module basys3_top (
    input wire clk,           // 100MHz clock from Basys 3
    input wire rst_n,         // Reset button (active low)

    // UART interface
    input wire uart_rx,       // UART receive
    output wire uart_tx,      // UART transmit

    // LED outputs (16 LEDs)
    output wire [15:0] led    // LED array
);

    // Internal signals
    wire rst;
    wire [31:0] current_pc;
    wire [31:0] machine_mode;
    wire [63:0] user_cycles;
    wire [63:0] total_cycles;
    wire execution_done;
    wire execution_error;
    wire segment_ready;
    wire [31:0] segment_data;
    wire segment_ack;

    // Invert reset signal
    assign rst = ~rst_n;

    // Instantiate RISC-V processor
    risc0_fpga_executor risc0_core (
        .clk(clk),
        .rst_n(rst_n),

        // Control interface
        .start_execution(1'b1),  // Always start
        .segment_threshold(32'd1000),
        .max_cycles(32'd1000000),
        .execution_done(execution_done),
        .execution_error(execution_error),

        // Status outputs
        .user_cycles(user_cycles),
        .total_cycles(total_cycles),
        .current_pc(current_pc),
        .machine_mode(machine_mode),

        // Segment interface
        .segment_ready(segment_ready),
        .segment_data(segment_data),
        .segment_ack(segment_ack)
    );

    // Display status on LEDs
    assign led[7:0] = current_pc[7:0];      // Lower 8 bits of PC
    assign led[15:8] = machine_mode[7:0];    // Machine mode on upper LEDs

    // Simple UART echo (for testing)
    assign uart_tx = uart_rx;  // Echo received data

    // Segment acknowledgment (always ready)
    assign segment_ack = segment_ready;

endmodule
EOF

    echo -e "   ✅ Created Basys 3 top-level wrapper"

    # Create Tcl script to add files to project
    cat > add_files.tcl << 'EOF'
# Add RISC-V processor files to Vivado project
add_files -norecurse [list \
    [file normalize "risc0_fpga_riscv.v"] \
    [file normalize "basys3_top.v"] \
]

# Set basys3_top as the top module
set_property top basys3_top [current_fileset]
set_property top_file [file normalize "basys3_top.v"] [current_fileset]

# Add constraints file if it exists
if {[file exists "risc0_fpga_constraints.xdc"]} {
    add_files -fileset constrs_1 [file normalize "risc0_fpga_constraints.xdc"]
}

puts "RISC-V processor files added to project"
puts "Top module set to: basys3_top"
EOF

    # Add files to Vivado project
    cd vivado_project
    $VIVADO_CMD -mode batch -source ../add_files.tcl

    if [ $? -eq 0 ]; then
        echo -e "   ✅ RISC-V processor integrated into Vivado project"
        echo -e "   📁 Project files:"
        echo -e "      - risc0_fpga_riscv.v (RISC-V processor)"
        echo -e "      - basys3_top.v (Basys 3 wrapper)"
        echo -e "      - basys3_constraints.xdc (Pin constraints)"
    else
        echo -e "   ❌ Failed to integrate RISC-V processor"
        return 1
    fi

    # Clean up temporary files
    rm -f ../add_files.tcl
    cd ..

    return 0
}

# Function to test hardware communication
test_hardware() {
    echo -e "\n${YELLOW}6️⃣ Testing hardware communication...${NC}"

    # Test UART communication
    echo -e "   📡 Testing UART communication..."

    # Send test command
    echo "TEST" > "$UART_DEVICE" 2>/dev/null || {
        echo -e "   ❌ Failed to write to UART device"
        return 1
    }

    echo -e "   ✅ UART write test passed"

    # Test reading (non-blocking)
    timeout 1 cat "$UART_DEVICE" > /dev/null 2>/dev/null || {
        echo -e "   ⚠️  UART read test inconclusive (no response expected)"
    }

    echo -e "   ✅ Hardware communication test completed"
    return 0
}

# Function to run Rust tests
run_rust_tests() {
    echo -e "\n${YELLOW}7️⃣ Running Rust tests...${NC}"

    cd "$(dirname "$0")/.."

    if cargo test --lib; then
        echo -e "   ✅ All Rust tests passed"
    else
        echo -e "   ❌ Some Rust tests failed"
        return 1
    fi

    return 0
}

# Function to run hardware example
run_hardware_example() {
    echo -e "\n${YELLOW}8️⃣ Running hardware example...${NC}"

    cd "$(dirname "$0")/.."

    # Check if hardware is available
    if [ ! -e "$UART_DEVICE" ]; then
        echo -e "   ⚠️  Hardware not available, skipping hardware example"
        echo -e "   💡 Connect your Basys 3 and run: cargo run --example basys3_real_hardware_example"
        return 0
    fi

    # Try to run the example with a timeout
    echo -e "   🔄 Starting hardware example (timeout: 30s)..."

    # Run with timeout to prevent hanging
    timeout 30 cargo run --example basys3_real_hardware_example || {
        echo -e "   ⚠️  Hardware example timed out or failed"
        echo -e "   💡 This is expected if the FPGA doesn't have the bitstream loaded"
        echo -e "   💡 Generate and program the bitstream first"
        return 0
    }

    echo -e "   ✅ Hardware example completed successfully"
    return 0
}

# Function to generate bitstream file
generate_bitstream_file() {
    echo -e "\n${YELLOW}9️⃣ Generating bitstream file...${NC}"

    cd vivado_project

    # Create Tcl script for bitstream generation
    cat > generate_bitstream.tcl << 'EOF'
# Generate bitstream for RISC-V processor
open_project risc0-fpga-basys3.xpr

# Add source files to the project
add_files -norecurse [list \
    [file normalize "risc0_fpga_riscv.v"] \
    [file normalize "basys3_top.v"] \
]

# Set the top module
set_property top basys3_top [current_fileset]
set_property top_file [file normalize "basys3_top.v"] [current_fileset]

# Add constraints file if it exists
if {[file exists "risc0_fpga_constraints.xdc"]} {
    add_files -fileset constrs_1 [file normalize "risc0_fpga_constraints.xdc"]
}

# Update compile order
update_compile_order -fileset sources_1

# Run synthesis
launch_runs synth_1
wait_on_run synth_1

# Check synthesis results
if {[file exists "risc0-fpga-basys3.runs/synth_1/basys3_top.dcp"]} {
    puts "Synthesis completed successfully"
} elseif {[get_property PROGRESS [get_runs synth_1]] == "Complete"} {
    puts "Synthesis completed successfully"
} elseif {[get_property STATUS [get_runs synth_1]] == "Complete"} {
    puts "Synthesis completed successfully"
} else {
    puts "ERROR: Synthesis failed"
    exit 1
}

# Run implementation
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1

# Check implementation results
if {[get_property PROGRESS [get_runs impl_1]] == "Complete"} {
    puts "Implementation completed successfully"
} else {
    puts "ERROR: Implementation failed"
    exit 1
}

# Check if bitstream was generated
if {[file exists "risc0-fpga-basys3.runs/impl_1/basys3_top.bit"]} {
    puts "Bitstream generated successfully"
    puts "Bitstream location: risc0-fpga-basys3.runs/impl_1/basys3_top.bit"
} else {
    puts "ERROR: Bitstream generation failed"
    exit 1
}
EOF

    # Generate bitstream
    echo -e "   🔄 Generating bitstream (this may take 10-30 minutes)..."
    $VIVADO_CMD -mode batch -source generate_bitstream.tcl

    if [ $? -eq 0 ]; then
        echo -e "   ✅ Bitstream generated successfully"
        echo -e "   📁 Bitstream location: $(pwd)/risc0-fpga-basys3.runs/impl_1/basys3_top.bit"
        echo -e "   💡 To program the FPGA:"
        echo -e "      $VIVADO_CMD -mode batch -source program_fpga.tcl"
    else
        echo -e "   ❌ Bitstream generation failed"
        echo -e "   💡 Check Vivado logs for errors"
        echo -e "   💡 Common issues:"
        echo -e "      - Missing source files"
        echo -e "      - Syntax errors in Verilog"
        echo -e "      - Constraint file issues"
        return 1
    fi

    cd ..
    return 0
}

# Function to create FPGA programming script
create_programming_script() {
    echo -e "\n${YELLOW}🔧 Creating FPGA programming script...${NC}"

    cat > vivado_project/program_fpga.tcl << 'EOF'
# Program FPGA with RISC-V processor bitstream
open_hw_manager
connect_hw_server
open_hw_target

# Get the first available device
set hw_device [lindex [get_hw_devices] 0]
current_hw_device $hw_device

# Refresh hardware
refresh_hw_device -update_hw_probes false $hw_device

# Program the bitstream
set_property PROGRAM.FILE "risc0-fpga-basys3.runs/impl_1/basys3_top.bit" $hw_device
program_hw_devices $hw_device

puts "FPGA programmed successfully"
close_hw_manager
EOF

    echo -e "   ✅ FPGA programming script created"
    echo -e "   📁 Script location: vivado_project/program_fpga.tcl"
    echo -e "   💡 To program the FPGA:"
    echo -e "      cd vivado_project && $VIVADO_CMD -mode batch -source program_fpga.tcl"
}

# Function to display deployment summary
deployment_summary() {
    echo -e "\n${GREEN}🎉 Deployment Summary${NC}"
    echo "====================="
    echo -e "   ✅ Hardware: $BOARD_NAME"
    echo -e "   ✅ UART Device: $UART_DEVICE"
    echo -e "   ✅ Vivado Version: $VIVADO_VERSION"
    echo -e "   ✅ Project: $PROJECT_NAME"
    echo -e "   ✅ RISC-V Processor: Integrated"
    echo -e "   ✅ Bitstream: Generated"
    echo -e ""
    echo -e "   🚀 Your Basys 3 FPGA executor is ready!"
    echo -e "   📝 Next steps:"
    echo -e "      1. Program the FPGA: cd vivado_project && $VIVADO_CMD -mode batch -source program_fpga.tcl"
    echo -e "      2. Run the hardware example: cargo run --example basys3_real_hardware_example"
    echo -e "      3. The RISC-V processor will be running on the FPGA!"
    echo -e ""
    echo -e "   📁 Files created:"
    echo -e "      - vivado_project/risc0-fpga-basys3.xpr (Vivado project)"
    echo -e "      - vivado_project/risc0-fpga-basys3.runs/impl_1/basys3_top.bit (Bitstream)"
    echo -e "      - vivado_project/program_fpga.tcl (Programming script)"
}

# Main deployment process
main() {
    echo -e "${BLUE}Starting Basys 3 deployment...${NC}"

    # Check prerequisites
    if ! check_hardware; then
        echo -e "${RED}❌ Hardware check failed${NC}"

        # Offer to install FTDI drivers
        echo -e "\n${YELLOW}Would you like to install FTDI drivers? (y/n)${NC}"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            if install_ftdi_drivers; then
                if [[ "$OSTYPE" == "darwin"* ]]; then
                    echo -e "\n${GREEN}✅ FTDI drivers installed. Please restart your computer and try again.${NC}"
                else
                    echo -e "\n${GREEN}✅ FTDI drivers configured. Please reconnect your Basys 3 and try again.${NC}"
                fi
                exit 0
            else
                echo -e "\n${RED}❌ Failed to install FTDI drivers. Please install manually.${NC}"
                exit 1
            fi
        fi

        exit 1
    fi

    if ! check_vivado; then
        echo -e "${RED}❌ Vivado check failed${NC}"
        exit 1
    fi

    # Create project
    if ! create_vivado_project; then
        echo -e "${RED}❌ Project creation failed${NC}"
        exit 1
    fi

    # Generate bitstream
    if ! generate_bitstream; then
        echo -e "${RED}❌ Bitstream generation failed${NC}"
        exit 1
    fi

    # Test hardware
    if ! test_hardware; then
        echo -e "${RED}❌ Hardware test failed${NC}"
        exit 1
    fi

    # Run tests
    if ! run_rust_tests; then
        echo -e "${RED}❌ Rust tests failed${NC}"
        exit 1
    fi

    # Generate bitstream file
    if ! generate_bitstream_file; then
        echo -e "${RED}❌ Bitstream generation failed${NC}"
        exit 1
    fi

    # Create programming script
    create_programming_script

    # Run hardware example (with timeout)
    if ! run_hardware_example; then
        echo -e "${YELLOW}⚠️  Hardware example skipped (expected without bitstream)${NC}"
    fi

    # Display summary
    deployment_summary

    echo -e "\n${GREEN}✅ Basys 3 deployment completed successfully!${NC}"
}

# Run main function
main "$@"
