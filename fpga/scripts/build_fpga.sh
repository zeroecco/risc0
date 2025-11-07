#!/bin/bash

# RISC0 FPGA Build Script
# Automates synthesis, implementation, and bitstream generation
# Copyright 2025 RISC Zero, Inc.

set -e

# Configuration
PROJECT_NAME="risc0_fpga"
DEVICE="xc7a35tcpg236-1"
TOP_MODULE="risc0_fpga_top"
CONSTRAINTS_FILE="risc0_fpga_constraints_fixed.xdc"
SOURCES=(
    "risc0_fpga_riscv_fixed.v"
    "risc0_fpga_top_fixed.v"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Vivado is available
check_vivado() {
    if ! command -v vivado &> /dev/null; then
        print_error "Vivado not found in PATH"
        print_error "Please install Xilinx Vivado and add it to your PATH"
        exit 1
    fi
    print_success "Vivado found: $(vivado -version | head -n1)"
}

# Function to create Vivado project
create_project() {
    print_status "Creating Vivado project..."

    vivado -mode batch -source - << EOF
create_project $PROJECT_NAME . -part $DEVICE -force
set_property board_part digilentinc:basys3:part0:1.2 [current_project]
EOF

    print_success "Project created successfully"
}

# Function to add source files
add_sources() {
    print_status "Adding source files..."

    for source in "${SOURCES[@]}"; do
        if [ -f "$source" ]; then
            vivado -mode batch -source - << EOF
add_files -norecurse $source
set_property file_type Verilog [get_files $source]
EOF
            print_success "Added $source"
        else
            print_error "Source file $source not found"
            exit 1
        fi
    done
}

# Function to add constraints
add_constraints() {
    print_status "Adding constraints file..."

    if [ -f "$CONSTRAINTS_FILE" ]; then
        vivado -mode batch -source - << EOF
add_files -fileset constrs_1 -norecurse $CONSTRAINTS_FILE
EOF
        print_success "Added constraints file"
    else
        print_error "Constraints file $CONSTRAINTS_FILE not found"
        exit 1
    fi
}

# Function to set top module
set_top_module() {
    print_status "Setting top module to $TOP_MODULE..."

    vivado -mode batch -source - << EOF
set_property top $TOP_MODULE [current_fileset]
set_property top_file [get_files $TOP_MODULE.v] [current_fileset]
EOF

    print_success "Top module set"
}

# Function to run synthesis
run_synthesis() {
    print_status "Running synthesis..."

    vivado -mode batch -source - << EOF
launch_runs synth_1
wait_on_run synth_1
EOF

    # Check synthesis results
    if [ -f "$PROJECT_NAME.runs/synth_1/runme.log" ]; then
        if grep -q "Synthesis completed successfully" "$PROJECT_NAME.runs/synth_1/runme.log"; then
            print_success "Synthesis completed successfully"
        else
            print_error "Synthesis failed"
            print_status "Check $PROJECT_NAME.runs/synth_1/runme.log for details"
            exit 1
        fi
    else
        print_error "Synthesis log not found"
        exit 1
    fi
}

# Function to run implementation
run_implementation() {
    print_status "Running implementation..."

    vivado -mode batch -source - << EOF
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1
EOF

    # Check implementation results
    if [ -f "$PROJECT_NAME.runs/impl_1/runme.log" ]; then
        if grep -q "Implementation completed successfully" "$PROJECT_NAME.runs/impl_1/runme.log"; then
            print_success "Implementation completed successfully"
        else
            print_error "Implementation failed"
            print_status "Check $PROJECT_NAME.runs/impl_1/runme.log for details"
            exit 1
        fi
    else
        print_error "Implementation log not found"
        exit 1
    fi
}

# Function to generate bitstream
generate_bitstream() {
    print_status "Generating bitstream..."

    if [ -f "$PROJECT_NAME.runs/impl_1/${PROJECT_NAME}_top.bit" ]; then
        cp "$PROJECT_NAME.runs/impl_1/${PROJECT_NAME}_top.bit" "./${PROJECT_NAME}.bit"
        print_success "Bitstream generated: ${PROJECT_NAME}.bit"
    else
        print_error "Bitstream not found"
        exit 1
    fi
}

# Function to run timing analysis
run_timing_analysis() {
    print_status "Running timing analysis..."

    vivado -mode batch -source - << EOF
open_run impl_1
report_timing_summary -file timing_report.txt
report_utilization -file utilization_report.txt
EOF

    if [ -f "timing_report.txt" ]; then
        print_success "Timing analysis completed"
        print_status "Check timing_report.txt for details"
    fi

    if [ -f "utilization_report.txt" ]; then
        print_success "Utilization analysis completed"
        print_status "Check utilization_report.txt for details"
    fi
}

# Function to run simulation
run_simulation() {
    print_status "Running simulation..."

    # Check if Icarus Verilog is available
    if command -v iverilog &> /dev/null; then
        print_status "Using Icarus Verilog for simulation"

        # Compile and run simulation
        iverilog -o sim_risc0_fpga risc0_fpga_testbench_fixed.v risc0_fpga_riscv_fixed.v risc0_fpga_top_fixed.v
        vvp sim_risc0_fpga

        print_success "Simulation completed"
    else
        print_warning "Icarus Verilog not found, skipping simulation"
        print_status "Install Icarus Verilog for simulation: sudo apt-get install iverilog"
    fi
}

# Function to clean up
cleanup() {
    print_status "Cleaning up temporary files..."
    rm -f sim_risc0_fpga
    print_success "Cleanup completed"
}

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -s, --simulate      Run simulation only"
    echo "  -c, --clean         Clean previous build"
    echo "  -t, --timing        Run timing analysis"
    echo "  -a, --all           Run complete build (default)"
    echo ""
    echo "Examples:"
    echo "  $0                  # Complete build"
    echo "  $0 -s              # Simulation only"
    echo "  $0 -c              # Clean and rebuild"
    echo "  $0 -t              # Build with timing analysis"
}

# Main function
main() {
    local simulate_only=false
    local clean_build=false
    local timing_analysis=false

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -s|--simulate)
                simulate_only=true
                shift
                ;;
            -c|--clean)
                clean_build=true
                shift
                ;;
            -t|--timing)
                timing_analysis=true
                shift
                ;;
            -a|--all)
                # Default behavior
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    print_status "Starting RISC0 FPGA build process..."

    # Check prerequisites
    check_vivado

    # Clean if requested
    if [ "$clean_build" = true ]; then
        print_status "Cleaning previous build..."
        rm -rf "$PROJECT_NAME" "$PROJECT_NAME.runs" "$PROJECT_NAME.xpr" "$PROJECT_NAME.bit"
        print_success "Clean completed"
    fi

    # Run simulation only if requested
    if [ "$simulate_only" = true ]; then
        run_simulation
        cleanup
        exit 0
    fi

    # Full build process
    create_project
    add_sources
    add_constraints
    set_top_module
    run_synthesis
    run_implementation
    generate_bitstream

    # Run timing analysis if requested
    if [ "$timing_analysis" = true ]; then
        run_timing_analysis
    fi

    # Run simulation
    run_simulation

    # Cleanup
    cleanup

    print_success "Build completed successfully!"
    print_status "Bitstream: ${PROJECT_NAME}.bit"
    print_status "Ready for programming to Basys 3 FPGA"
}

# Run main function with all arguments
main "$@"

