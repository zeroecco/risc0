#!/bin/bash

# ============================================================================
# SIMPLE UART ECHO TEST FOR BASYS 3
# ============================================================================
# This creates a minimal design that only implements UART echo

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "🚀 Simple UART Echo Test for Basys 3"
echo -e "====================================="

# Check Vivado
VIVADO_CMD="/tools/xilinx/2025.1/Vivado/bin/vivado"
if [ ! -f "$VIVADO_CMD" ]; then
    echo -e "${RED}❌ Vivado not found at $VIVADO_CMD${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Vivado found${NC}"

# Create project directory
mkdir -p uart_test_project
cd uart_test_project

# Create simple UART echo top-level module
cat > uart_echo_top.v << 'EOF'
// Simple UART Echo for Basys 3
// Just echoes received data back

module uart_echo_top (
    input wire clk,           // 100MHz clock
    input wire rst_n,         // Reset button (active low)
    input wire uart_rx,       // UART receive
    output wire uart_tx,      // UART transmit
    output wire [15:0] led    // LED array for status
);

    // Simple UART echo
    assign uart_tx = uart_rx;  // Echo received data

    // Show activity on LEDs
    reg [15:0] led_counter;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            led_counter <= 16'h0;
        end else begin
            led_counter <= led_counter + 1;
        end
    end

    assign led = led_counter;

endmodule
EOF

# Create constraints file
cat > uart_echo_constraints.xdc << 'EOF'
# Basys 3 UART Echo Constraints

# Clock
set_property PACKAGE_PIN W5 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]

# Reset
set_property PACKAGE_PIN U18 [get_ports rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports rst_n]

# UART
set_property PACKAGE_PIN A18 [get_ports uart_rx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_rx]

set_property PACKAGE_PIN B18 [get_ports uart_tx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_tx]

# LEDs
set_property PACKAGE_PIN U16 [get_ports {led[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[0]}]

set_property PACKAGE_PIN E19 [get_ports {led[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[1]}]

set_property PACKAGE_PIN U19 [get_ports {led[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[2]}]

set_property PACKAGE_PIN V19 [get_ports {led[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[3]}]

set_property PACKAGE_PIN W18 [get_ports {led[4]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[4]}]

set_property PACKAGE_PIN U15 [get_ports {led[5]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[5]}]

set_property PACKAGE_PIN U14 [get_ports {led[6]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[6]}]

set_property PACKAGE_PIN V14 [get_ports {led[7]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[7]}]

set_property PACKAGE_PIN V13 [get_ports {led[8]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[8]}]

set_property PACKAGE_PIN V3 [get_ports {led[9]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[9]}]

set_property PACKAGE_PIN W3 [get_ports {led[10]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[10]}]

set_property PACKAGE_PIN U3 [get_ports {led[11]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[11]}]

set_property PACKAGE_PIN P3 [get_ports {led[12]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[12]}]

set_property PACKAGE_PIN N3 [get_ports {led[13]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[13]}]

set_property PACKAGE_PIN P1 [get_ports {led[14]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[14]}]

set_property PACKAGE_PIN L1 [get_ports {led[15]}]
set_property IOSTANDARD LVCMOS33 [get_ports {led[15]}]

# Timing constraints
set_false_path -from [get_ports clk]
set_false_path -to [get_ports led*]
set_false_path -from [get_ports uart_rx]
set_false_path -to [get_ports uart_tx]
EOF

echo -e "${GREEN}✅ Created simple UART echo design${NC}"

# Create Vivado project
echo -e "\n${YELLOW}🔄 Creating Vivado project...${NC}"
cat > create_project.tcl << 'EOF'
create_project uart-echo-test . -part xc7a35tcpg236-1 -force
# Try to set board part, but continue if it fails
catch {set_property board_part digilentinc:basys3:part0:1.2 [current_project]}
EOF

$VIVADO_CMD -mode batch -source create_project.tcl

# Add files to project
cat > add_files.tcl << 'EOF'
# Add source files
add_files -norecurse [list \
    [file normalize "uart_echo_top.v"] \
]

# Set top module for the project
set_property top uart_echo_top [current_project]
set_property top_file [file normalize "uart_echo_top.v"] [current_project]

# Set top module for the fileset
set_property top uart_echo_top [current_fileset]
set_property top_file [file normalize "uart_echo_top.v"] [current_fileset]

# Add constraints
add_files -fileset constrs_1 [file normalize "uart_echo_constraints.xdc"]

# Verify top module is set
puts "Files added to project"
puts "Project top module: [get_property top [current_project]]"
puts "Fileset top module: [get_property top [current_fileset]]"
puts "Top module file: [get_property top_file [current_fileset]]"
EOF

$VIVADO_CMD -mode batch -source add_files.tcl

# Generate bitstream
echo -e "\n${YELLOW}🔄 Generating bitstream...${NC}"
cat > generate_bitstream.tcl << 'EOF'
open_project uart-echo-test.xpr

# Verify top module is set
puts "Project top module: [get_property top [current_project]]"
puts "Fileset top module: [get_property top [current_fileset]]"
puts "Current top file: [get_property top_file [current_fileset]]"

# Set top module again if needed
if {[get_property top [current_fileset]] == ""} {
    set_property top uart_echo_top [current_fileset]
    set_property top_file [file normalize "uart_echo_top.v"] [current_fileset]
    puts "Re-set top module to: uart_echo_top"
}

# Run synthesis
launch_runs synth_1
wait_on_run synth_1

# Run implementation
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1

puts "Bitstream generation completed"
EOF

$VIVADO_CMD -mode batch -source generate_bitstream.tcl

echo -e "${GREEN}✅ Bitstream generated successfully!${NC}"
echo -e "${GREEN}📁 Bitstream location: uart_test_project/uart-echo-test.runs/impl_1/uart_echo_top.bit${NC}"

# Create programming script
cat > program_uart_echo.tcl << 'EOF'
# Program UART Echo Design
open_hw_manager
connect_hw_server

set hw_targets [get_hw_targets]
set hw_target [lindex $hw_targets 0]
open_hw_target $hw_target

set hw_devices [get_hw_devices]
set hw_device [lindex $hw_devices 0]
current_hw_device $hw_device

set_property PROGRAM.FILE "uart-echo-test.runs/impl_1/uart_echo_top.bit" $hw_device
program_hw_devices $hw_device

puts "UART echo design programmed successfully!"
close_hw_manager
EOF

echo -e "\n${GREEN}✅ UART echo test ready!${NC}"
echo -e "${YELLOW}To program:${NC}"
echo -e "   cd uart_test_project"
echo -e "   /tools/xilinx/2025.1/Vivado/bin/vivado -mode batch -source program_uart_echo.tcl"
echo -e ""
echo -e "${YELLOW}After programming:${NC}"
echo -e "   cargo run --bin test_uart"
