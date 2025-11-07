// RISC0 FPGA Testbench
// Testbench for the RISC-V executor and accelerators
// Copyright 2025 RISC Zero, Inc.

`timescale 1ns / 1ps

module risc0_fpga_testbench;

    // Clock and reset
    reg clk;
    reg rst_n;

    // Test signals
    reg [31:0] test_addr;
    reg [31:0] test_data_in;
    wire [31:0] test_data_out;
    reg test_we, test_re;
    wire test_ready;

    // Control signals
    reg start_execution;
    reg [31:0] segment_threshold;
    reg [31:0] max_cycles;
    wire execution_done;
    wire execution_error;

    // Status signals
    wire [63:0] user_cycles;
    wire [63:0] total_cycles;
    wire [31:0] current_pc;

    // Instantiate the top module
    risc0_fpga_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .host_addr(test_addr),
        .host_data_in(test_data_in),
        .host_data_out(test_data_out),
        .host_we(test_we),
        .host_re(test_re),
        .host_ready(test_ready),
        .start_execution(start_execution),
        .segment_threshold(segment_threshold),
        .max_cycles(max_cycles),
        .execution_done(execution_done),
        .execution_error(execution_error),
        .user_cycles(user_cycles),
        .total_cycles(total_cycles),
        .current_pc(current_pc)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // 100MHz clock
    end

    // Test stimulus
    initial begin
        // Initialize signals
        rst_n = 0;
        start_execution = 0;
        test_addr = 32'h0;
        test_data_in = 32'h0;
        test_we = 0;
        test_re = 0;
        segment_threshold = 1000;
        max_cycles = 10000;

        // Reset
        #100;
        rst_n = 1;
        #50;

        // Load test program into memory
        load_test_program();

        // Start execution
        start_execution = 1;
        #10;
        start_execution = 0;

        // Wait for completion
        wait(execution_done || execution_error);

        // Display results
        $display("Execution completed:");
        $display("User cycles: %d", user_cycles);
        $display("Total cycles: %d", total_cycles);
        $display("Final PC: 0x%h", current_pc);

        if (execution_error)
            $display("ERROR: Execution failed");
        else
            $display("SUCCESS: Execution completed successfully");

        #100;
        $finish;
    end

    // Task to load test program
    task load_test_program;
        begin
            // Simple test program: addi, add, ecall
            write_memory(32'h0, 32'h00100093);  // addi x1, x0, 1
            write_memory(32'h4, 32'h00200113);  // addi x2, x0, 2
            write_memory(32'h8, 32'h00208133);  // add x2, x1, x2
            write_memory(32'hc, 32'h00000073);  // ecall
        end
    endtask

    // Task to write to memory
    task write_memory;
        input [31:0] addr;
        input [31:0] data;
        begin
            @(posedge clk);
            test_addr = addr;
            test_data_in = data;
            test_we = 1;
            @(posedge clk);
            test_we = 0;
        end
    endtask

    // Monitor execution
    always @(posedge clk) begin
        if (execution_done || execution_error) begin
            $display("Simulation finished at time %t", $time);
        end
    end

endmodule
