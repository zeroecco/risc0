# Basys 3 Artix-7 FPGA Constraints for RISC0 RISC-V Processor
# Fixed implementation with proper pin assignments and timing constraints

# ============================================================================
# CLOCK CONSTRAINTS
# ============================================================================

# Clock signal (100MHz)
set_property PACKAGE_PIN W5 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -period 10.000 -name clk -waveform {0.000 5.000} [get_ports clk]

# Clock domain constraints
set_clock_groups -asynchronous -group [get_clocks clk]

# ============================================================================
# RESET AND CONTROL SIGNALS
# ============================================================================

# Reset signal (active low)
set_property PACKAGE_PIN U18 [get_ports rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports rst_n]

# Start execution signal
set_property PACKAGE_PIN V17 [get_ports start_execution]
set_property IOSTANDARD LVCMOS33 [get_ports start_execution]

# ============================================================================
# UART COMMUNICATION INTERFACE
# ============================================================================

# UART signals for host communication
set_property PACKAGE_PIN B18 [get_ports uart_tx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_tx]
set_property PACKAGE_PIN A18 [get_ports uart_rx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_rx]

# UART clock domain
create_clock -period 8680.555 -name uart_clk -waveform {0.000 4340.278} [get_ports uart_rx]

# ============================================================================
# STATUS LEDS
# ============================================================================

# LED signals for status indication
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

# ============================================================================
# SWITCHES FOR CONFIGURATION
# ============================================================================

# Configuration switches
set_property PACKAGE_PIN V17 [get_ports {sw[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {sw[0]}]
set_property PACKAGE_PIN V16 [get_ports {sw[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {sw[1]}]
set_property PACKAGE_PIN W16 [get_ports {sw[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {sw[2]}]
set_property PACKAGE_PIN W17 [get_ports {sw[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {sw[3]}]
set_property PACKAGE_PIN W15 [get_ports {sw[4]}]
set_property IOSTANDARD LVCMOS33 [get_ports {sw[4]}]
set_property PACKAGE_PIN V15 [get_ports {sw[5]}]
set_property IOSTANDARD LVCMOS33 [get_ports {sw[5]}]
set_property PACKAGE_PIN W14 [get_ports {sw[6]}]
set_property IOSTANDARD LVCMOS33 [get_ports {sw[6]}]
set_property PACKAGE_PIN W13 [get_ports {sw[7]}]
set_property IOSTANDARD LVCMOS33 [get_ports {sw[7]}]

# ============================================================================
# BUTTONS FOR CONTROL
# ============================================================================

# Control buttons
set_property PACKAGE_PIN U18 [get_ports {btn[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {btn[0]}]
set_property PACKAGE_PIN T18 [get_ports {btn[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {btn[1]}]
set_property PACKAGE_PIN W19 [get_ports {btn[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {btn[2]}]
set_property PACKAGE_PIN T17 [get_ports {btn[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {btn[3]}]

# ============================================================================
# TIMING CONSTRAINTS
# ============================================================================

# Set false paths for asynchronous inputs
set_false_path -from [get_ports rst_n]
set_false_path -from [get_ports start_execution]
set_false_path -from [get_ports {btn[*]}]
set_false_path -from [get_ports {sw[*]}]

# Set false paths for outputs
set_false_path -to [get_ports led*]
set_false_path -to [get_ports uart_tx]

# Clock domain constraints
set_clock_groups -asynchronous -group [get_clocks clk] -group [get_clocks uart_clk]

# Setup and hold time constraints for UART
set_input_delay -clock uart_clk -max 1.0 [get_ports uart_rx]
set_output_delay -clock uart_clk -max 1.0 [get_ports uart_tx]

# Memory timing constraints
set_max_delay -from [get_clocks clk] -to [get_clocks clk] 8.0
set_min_delay -from [get_clocks clk] -to [get_clocks clk] 0.5

# Multi-cycle paths for memory access
set_multicycle_path -setup 2 -from [get_clocks clk] -to [get_clocks clk] -through [get_pins */memory*]
set_multicycle_path -hold 1 -from [get_clocks clk] -to [get_clocks clk] -through [get_pins */memory*]

# ============================================================================
# SYNTHESIS CONSTRAINTS
# ============================================================================

# BRAM inference for memory
set_property RAM_STYLE block [get_cells -hierarchical -filter {NAME =~ "*memory*"}]

# Register inference for state machines
set_property FSM_ENCODING gray [get_cells -hierarchical -filter {NAME =~ "*exec_state*"}]

# Optimize for performance
set_property OPTIMIZATION_MODE Performance [current_design]

# ============================================================================
# IMPLEMENTATION CONSTRAINTS
# ============================================================================

# Placement constraints for critical paths
set_property LOC SLICE_X0Y0 [get_cells -hierarchical -filter {NAME =~ "*pc*"}]
set_property LOC SLICE_X0Y1 [get_cells -hierarchical -filter {NAME =~ "*registers*"}]

# Clock gating (if available)
set_property CLOCK_GATE true [get_cells -hierarchical -filter {NAME =~ "*exec_state*"}]

# ============================================================================
# DEBUG CONSTRAINTS
# ============================================================================

# Debug signals (can be connected to LEDs for debugging)
set_property MARK_DEBUG true [get_nets -hierarchical -filter {NAME =~ "*current_pc*"}]
set_property MARK_DEBUG true [get_nets -hierarchical -filter {NAME =~ "*execution_done*"}]
set_property MARK_DEBUG true [get_nets -hierarchical -filter {NAME =~ "*execution_error*"}]

# ============================================================================
# POWER CONSTRAINTS
# ============================================================================

# Power optimization
set_property POWER_OPTIMIZATION true [current_design]

# Clock gating for power reduction
set_property CLOCK_GATE true [get_cells -hierarchical -filter {NAME =~ "*idle*"}]

# ============================================================================
# AREA CONSTRAINTS
# ============================================================================

# Area optimization for BRAM usage
set_property RAM_STYLE block [get_cells -hierarchical -filter {NAME =~ "*memory*"}]

# DSP usage for arithmetic operations
set_property USE_DSP true [get_cells -hierarchical -filter {NAME =~ "*alu*"}]

# ============================================================================
# VERIFICATION CONSTRAINTS
# ============================================================================

# Formal verification properties
set_property VERILOG_MACRO "FORMAL=1" [current_design]

# Coverage constraints
set_property COVERAGE true [current_design]

# ============================================================================
# SIMULATION CONSTRAINTS
# ============================================================================

# Simulation time scale
set_property SIMULATION_TIME_SCALE "1ns/1ps" [current_design]

# Waveform generation
set_property WAVEFORM_GENERATION true [current_design]

# ============================================================================
# DOCUMENTATION
# ============================================================================

# Pin assignments documentation
# LED[0]: Execution active
# LED[1]: Memory access
# LED[2]: Error state
# LED[3]: Segment ready
# LED[4-7]: Current PC bits [3:0]
# LED[8-11]: User cycles bits [3:0]
# LED[12-15]: Status bits

# Switch assignments
# SW[0]: Debug mode
# SW[1]: Verbose output
# SW[2]: Performance mode
# SW[3]: Memory test mode
# SW[4-7]: Reserved

# Button assignments
# BTN[0]: Reset
# BTN[1]: Start execution
# BTN[2]: Step execution
# BTN[3]: Clear error

