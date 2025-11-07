# ============================================================================
# FPGA Programming Script for Basys 3
# ============================================================================
# This script programs the Basys 3 FPGA with the RISC-V processor bitstream

puts "🚀 Programming Basys 3 FPGA with RISC-V Processor"
puts "=================================================="

# Open hardware manager
open_hw_manager

# Connect to hardware
connect_hw_server

# Get the first available hardware target
set hw_targets [get_hw_targets]
if {[llength $hw_targets] == 0} {
    puts "❌ No hardware targets found"
    puts "💡 Please check:"
    puts "   1. Basys 3 is connected via USB"
    puts "   2. Board is powered on"
    puts "   3. FTDI drivers are installed"
    exit 1
}

set hw_target [lindex $hw_targets 0]
puts "✅ Found hardware target: $hw_target"

# Open the hardware target
open_hw_target $hw_target

# Get the first available device
set hw_devices [get_hw_devices]
if {[llength $hw_devices] == 0} {
    puts "❌ No hardware devices found"
    puts "💡 Please check if Basys 3 is properly connected"
    exit 1
}

set hw_device [lindex $hw_devices 0]
puts "✅ Found hardware device: $hw_device"

# Set the programming file
set bitstream_file "risc0-fpga-basys3.runs/impl_1/basys3_top.bit"

if {![file exists $bitstream_file]} {
    puts "❌ Bitstream file not found: $bitstream_file"
    puts "💡 Please run the deployment script first"
    exit 1
}

puts "✅ Found bitstream file: $bitstream_file"

# Program the FPGA
puts "🔄 Programming FPGA (this may take a few seconds)..."
current_hw_device $hw_device
refresh_hw_device -update_hw_probes false $hw_device

# Set the programming file
set_property PROGRAM.FILE $bitstream_file $hw_device

# Program the device
program_hw_devices $hw_device

puts "✅ FPGA programming completed successfully!"
puts ""
puts "🎉 RISC-V Processor is now running on Basys 3!"
puts ""
puts "📋 What to expect:"
puts "   • LEDs[7:0] = Current Program Counter (PC)"
puts "   • LEDs[15:8] = Machine Mode"
puts "   • UART echo = Any data sent will be echoed back"
puts ""
puts "🔧 To test:"
puts "   • Watch the LEDs change as the processor executes"
puts "   • Connect via UART to test serial communication"
puts "   • Press reset button to restart the processor"
puts ""

# Close hardware manager
close_hw_manager

puts "✅ Programming script completed"
