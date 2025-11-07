// RISC0 FPGA Testbench - Fixed Implementation
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

        // Test 1: Basic arithmetic
        $display("=== Test 1: Basic Arithmetic ===");
        load_arithmetic_test();
        run_test("Arithmetic Test");

        // Test 2: Memory operations
        $display("=== Test 2: Memory Operations ===");
        load_memory_test();
        run_test("Memory Test");

        // Test 3: Branch and jump
        $display("=== Test 3: Branch and Jump ===");
        load_branch_test();
        run_test("Branch Test");

        // Test 4: Complex program
        $display("=== Test 4: Complex Program ===");
        load_complex_test();
        run_test("Complex Test");

        #100;
        $display("All tests completed!");
        $finish;
    end

    // Task to run a test
    task run_test;
        input [80*8-1:0] test_name;
        begin
            $display("Running %s...", test_name);
            
            // Start execution
            start_execution = 1;
            #10;
            start_execution = 0;

            // Wait for completion
            wait(execution_done || execution_error);

            // Display results
            $display("Test completed:");
            $display("  User cycles: %d", user_cycles);
            $display("  Total cycles: %d", total_cycles);
            $display("  Final PC: 0x%h", current_pc);

            if (execution_error)
                $display("  ERROR: Execution failed");
            else
                $display("  SUCCESS: Execution completed successfully");

            // Reset for next test
            #50;
            rst_n = 0;
            #10;
            rst_n = 1;
            #50;
        end
    endtask

    // Task to load arithmetic test program
    task load_arithmetic_test;
        begin
            // Test program: addi, add, sub, xor, or, and
            write_memory(32'h0, 32'h00100093);  // addi x1, x0, 1
            write_memory(32'h4, 32'h00200113);  // addi x2, x0, 2
            write_memory(32'h8, 32'h00300193);  // addi x3, x0, 3
            write_memory(32'hc, 32'h00208133);  // add x2, x1, x2
            write_memory(32'h10, 32'h40310233); // sub x4, x2, x3
            write_memory(32'h14, 32'h0041e333); // xor x6, x3, x4
            write_memory(32'h18, 32'h0061f333); // or x6, x3, x6
            write_memory(32'h1c, 32'h0061e333); // and x6, x3, x6
            write_memory(32'h20, 32'h00000073); // ecall
        end
    endtask

    // Task to load memory test program
    task load_memory_test;
        begin
            // Test program: lui, sw, lw
            write_memory(32'h0, 32'h00001137);  // lui x2, 1
            write_memory(32'h4, 32'h00400113);  // addi x2, x0, 4
            write_memory(32'h8, 32'h00500193);  // addi x3, x0, 5
            write_memory(32'hc, 32'h00312023);  // sw x3, 0(x2)
            write_memory(32'h10, 32'h00012283); // lw x5, 0(x2)
            write_memory(32'h14, 32'h00000073); // ecall
        end
    endtask

    // Task to load branch test program
    task load_branch_test;
        begin
            // Test program: beq, bne, jal
            write_memory(32'h0, 32'h00100093);  // addi x1, x0, 1
            write_memory(32'h4, 32'h00100113);  // addi x2, x0, 1
            write_memory(32'h8, 32'h00208163);  // beq x1, x2, 2
            write_memory(32'hc, 32'h00300193);  // addi x3, x0, 3
            write_memory(32'h10, 32'h00208163); // beq x1, x2, 2
            write_memory(32'h14, 32'h00400213); // addi x4, x0, 4
            write_memory(32'h18, 32'h00400293); // addi x5, x0, 4
            write_memory(32'h1c, 32'h00521663); // bne x4, x5, 12
            write_memory(32'h20, 32'h00600313); // addi x6, x0, 6
            write_memory(32'h24, 32'h0100006f); // jal x0, 16
            write_memory(32'h28, 32'h00700393); // addi x7, x0, 7
            write_memory(32'h2c, 32'h00000073); // ecall
        end
    endtask

    // Task to load complex test program
    task load_complex_test;
        begin
            // Complex test: arithmetic, memory, and control flow
            write_memory(32'h0, 32'h00100093);  // addi x1, x0, 1
            write_memory(32'h4, 32'h00200113);  // addi x2, x0, 2
            write_memory(32'h8, 32'h00300193);  // addi x3, x0, 3
            write_memory(32'hc, 32'h00400213);  // addi x4, x0, 4
            write_memory(32'h10, 32'h00500293); // addi x5, x0, 5
            write_memory(32'h14, 32'h00208133); // add x2, x1, x2
            write_memory(32'h18, 32'h00310233); // add x4, x2, x3
            write_memory(32'h1c, 32'h00418333); // add x6, x3, x4
            write_memory(32'h20, 32'h00602023); // sw x6, 0(x0)
            write_memory(32'h24, 32'h00002283); // lw x5, 0(x0)
            write_memory(32'h28, 32'h00508163); // beq x1, x5, 2
            write_memory(32'h2c, 32'h00600313); // addi x6, x0, 6
            write_memory(32'h30, 32'h00600393); // addi x7, x0, 6
            write_memory(32'h34, 32'h00731663); // bne x6, x7, 12
            write_memory(32'h38, 32'h00800413); // addi x8, x0, 8
            write_memory(32'h3c, 32'h0100006f); // jal x0, 16
            write_memory(32'h40, 32'h00900493); // addi x9, x0, 9
            write_memory(32'h44, 32'h00000073); // ecall
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

    // Monitor memory writes
    always @(posedge clk) begin
        if (test_we) begin
            $display("Memory write: addr=0x%h, data=0x%h", test_addr, test_data_in);
        end
    end

endmodule


