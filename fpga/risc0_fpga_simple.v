// RISC0 FPGA Simple RISC-V Executor
// Simplified version compatible with Icarus Verilog
// Copyright 2025 RISC Zero, Inc.

`timescale 1ns / 1ps

module risc0_fpga_simple (
    input wire clk,
    input wire rst_n,

    // Memory interface
    input wire [31:0] mem_addr,
    input wire [31:0] mem_data_in,
    output wire [31:0] mem_data_out,
    input wire mem_we,
    input wire mem_re,
    output wire mem_ready,

    // Control interface
    input wire start_execution,
    input wire [31:0] segment_threshold,
    input wire [31:0] max_cycles,
    output wire execution_done,
    output wire execution_error,

    // Status outputs
    output wire [63:0] user_cycles,
    output wire [63:0] total_cycles,
    output wire [31:0] current_pc,

    // Segment interface
    output wire segment_ready,
    output wire [31:0] segment_data,
    input wire segment_ack
);

    // Internal registers
    reg [2:0] exec_state;
    reg [31:0] pc;
    reg [63:0] user_cycles_reg;
    reg [63:0] total_cycles_reg;
    reg [31:0] segment_counter;
    reg [31:0] segment_cycles;
    reg segment_ready_reg;
    reg [31:0] segment_data_reg;
    reg execution_error_reg;
    reg execution_done_reg;

    // Memory and registers
    reg [31:0] registers [0:31];
    reg [31:0] memory [0:1023];

    // Instruction execution
    reg [31:0] current_instruction;
    reg [7:0] instruction_type;

    // State definitions
    parameter IDLE = 3'b000;
    parameter FETCH = 3'b001;
    parameter DECODE = 3'b010;
    parameter EXECUTE = 3'b011;
    parameter MEMORY = 3'b100;
    parameter SEGMENT_CHECK = 3'b110;
    parameter ERROR = 3'b111;

    // Instruction types
    parameter INSN_ADD = 8'h00;
    parameter INSN_SUB = 8'h01;
    parameter INSN_ADDI = 8'h07;
    parameter INSN_LUI = 8'h15;
    parameter INSN_LW = 8'h2A;
    parameter INSN_SW = 8'h32;
    parameter INSN_BEQ = 8'h0D;
    parameter INSN_JAL = 8'h13;
    parameter INSN_EANY = 8'h38;
    parameter INSN_INVALID = 8'hFF;

    // Loop variable for initialization
    integer i;

    // Main execution state machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            exec_state <= IDLE;
            pc <= 32'h0;
            user_cycles_reg <= 64'h0;
            total_cycles_reg <= 64'h0;
            segment_counter <= 32'h0;
            segment_cycles <= 32'h0;
            segment_ready_reg <= 1'b0;
            segment_data_reg <= 32'h0;
            execution_error_reg <= 1'b0;
            execution_done_reg <= 1'b0;

            // Initialize registers
            for (i = 0; i < 32; i = i + 1) begin
                registers[i] <= 32'h0;
            end

            // Initialize memory
            for (i = 0; i < 1024; i = i + 1) begin
                memory[i] <= 32'h0;
            end
        end else begin
            case (exec_state)
                IDLE: begin
                    if (start_execution) begin
                        $display("Starting execution at time %t", $time);
                        exec_state <= FETCH;
                        pc <= 32'h0;
                        user_cycles_reg <= 64'h0;
                        total_cycles_reg <= 64'h0;
                        segment_counter <= 32'h0;
                        segment_cycles <= 32'h0;
                        execution_error_reg <= 1'b0;
                        execution_done_reg <= 1'b0;
                    end else begin
                        $display("IDLE: waiting for start_execution, current=%b", start_execution);
                    end
                end

                FETCH: begin
                    $display("FETCH: PC=0x%h, instruction=0x%h", pc, memory[pc[11:2]]);
                    current_instruction <= memory[pc[11:2]];
                    exec_state <= DECODE;
                    total_cycles_reg <= total_cycles_reg + 1;
                end

                DECODE: begin
                    instruction_type <= decode_instruction(current_instruction);
                    $display("DECODE: instruction=0x%h, type=0x%h", current_instruction, decode_instruction(current_instruction));
                    exec_state <= EXECUTE;
                end

                EXECUTE: begin
                    case (instruction_type)
                        INSN_ADD: begin
                            $display("EXECUTE: ADD");
                            registers[get_rd(current_instruction)] <=
                                registers[get_rs1(current_instruction)] +
                                registers[get_rs2(current_instruction)];
                            pc <= pc + 4;
                            exec_state <= SEGMENT_CHECK;
                        end

                        INSN_SUB: begin
                            $display("EXECUTE: SUB");
                            registers[get_rd(current_instruction)] <=
                                registers[get_rs1(current_instruction)] -
                                registers[get_rs2(current_instruction)];
                            pc <= pc + 4;
                            exec_state <= SEGMENT_CHECK;
                        end

                        INSN_ADDI: begin
                            $display("EXECUTE: ADDI - PC before: 0x%h, PC after: 0x%h", pc, pc + 4);
                            registers[get_rd(current_instruction)] <=
                                registers[get_rs1(current_instruction)] +
                                get_imm_i(current_instruction);
                            pc <= pc + 4;
                            exec_state <= SEGMENT_CHECK;
                        end

                        INSN_LUI: begin
                            $display("EXECUTE: LUI");
                            registers[get_rd(current_instruction)] <=
                                get_imm_u(current_instruction);
                            pc <= pc + 4;
                            exec_state <= SEGMENT_CHECK;
                        end

                        INSN_LW: begin
                            $display("EXECUTE: LW");
                            exec_state <= MEMORY;
                        end

                        INSN_SW: begin
                            $display("EXECUTE: SW");
                            exec_state <= MEMORY;
                        end

                        INSN_BEQ: begin
                            $display("EXECUTE: BEQ");
                            if (registers[get_rs1(current_instruction)] ==
                                registers[get_rs2(current_instruction)]) begin
                                pc <= pc + get_imm_b(current_instruction);
                            end else begin
                                pc <= pc + 4;
                            end
                            exec_state <= SEGMENT_CHECK;
                        end

                        INSN_JAL: begin
                            $display("EXECUTE: JAL");
                            registers[get_rd(current_instruction)] <= pc + 4;
                            pc <= pc + get_imm_j(current_instruction);
                            exec_state <= SEGMENT_CHECK;
                        end

                        INSN_EANY: begin
                            $display("EXECUTE: ECALL - terminating");
                            // System call - terminate execution
                            execution_done_reg <= 1'b1;
                            exec_state <= IDLE;
                        end

                        default: begin
                            $display("EXECUTE: INVALID instruction 0x%h", instruction_type);
                            execution_error_reg <= 1'b1;
                            exec_state <= ERROR;
                        end
                    endcase

                    user_cycles_reg <= user_cycles_reg + 1;
                    segment_cycles <= segment_cycles + 1;
                end

                MEMORY: begin
                    case (instruction_type)
                        INSN_LW: begin
                            $display("MEMORY: LW");
                            registers[get_rd(current_instruction)] <=
                                memory[get_rs1(current_instruction) + get_imm_i(current_instruction)];
                        end

                        INSN_SW: begin
                            $display("MEMORY: SW");
                            memory[get_rs1(current_instruction) + get_imm_s(current_instruction)] <=
                                registers[get_rs2(current_instruction)];
                        end
                    endcase

                    pc <= pc + 4;
                    exec_state <= SEGMENT_CHECK;
                end

                SEGMENT_CHECK: begin
                    if (segment_cycles >= segment_threshold) begin
                        $display("SEGMENT_CHECK: Creating segment");
                        segment_ready_reg <= 1'b1;
                        segment_data_reg <= segment_counter;
                        segment_counter <= segment_counter + 1;
                        segment_cycles <= 32'h0;

                        if (segment_ack) begin
                            segment_ready_reg <= 1'b0;
                            exec_state <= FETCH;
                        end
                    end else if (total_cycles_reg >= max_cycles) begin
                        $display("SEGMENT_CHECK: Max cycles reached");
                        execution_done_reg <= 1'b1;
                        exec_state <= IDLE;
                    end else begin
                        exec_state <= FETCH;
                    end
                end

                ERROR: begin
                    $display("ERROR: Execution error");
                    execution_error_reg <= 1'b1;
                    execution_done_reg <= 1'b1;
                    exec_state <= IDLE;
                end

                default: begin
                    exec_state <= IDLE;
                end
            endcase
        end
    end

    // Instruction decoder function
    function [7:0] decode_instruction;
        input [31:0] instruction;
        reg [6:0] opcode;
        reg [2:0] func3;
        reg [6:0] func7;
    begin
        opcode = instruction[6:0];
        func3 = instruction[14:12];
        func7 = instruction[31:25];

        $display("DECODE_DEBUG: opcode=0x%h, func3=0x%h, func7=0x%h", opcode, func3, func7);

        case (opcode)
            7'h33: begin  // R-format arithmetic
                case ({func3, func7})
                    10'h000: decode_instruction = INSN_ADD;
                    10'h020: decode_instruction = INSN_SUB;
                    default: decode_instruction = INSN_INVALID;
                endcase
            end

            7'h13: begin  // I-format arithmetic
                case (func3)
                    3'h0: decode_instruction = INSN_ADDI;
                    default: decode_instruction = INSN_INVALID;
                endcase
            end

            7'h03: begin  // I-format loads
                case (func3)
                    3'h2: decode_instruction = INSN_LW;
                    default: decode_instruction = INSN_INVALID;
                endcase
            end

            7'h23: begin  // S-format stores
                case (func3)
                    3'h2: decode_instruction = INSN_SW;
                    default: decode_instruction = INSN_INVALID;
                endcase
            end

            7'h37: decode_instruction = INSN_LUI;    // U-format lui

            7'h63: begin  // B-format branches
                case (func3)
                    3'h0: decode_instruction = INSN_BEQ;
                    default: decode_instruction = INSN_INVALID;
                endcase
            end

            7'h6F: decode_instruction = INSN_JAL;   // J-format jal

            7'h73: begin  // System instructions
                case ({func3, func7})
                    10'h000: decode_instruction = INSN_EANY;
                    default: decode_instruction = INSN_INVALID;
                endcase
            end

            default: decode_instruction = INSN_INVALID;
        endcase
    end
    endfunction

    // Instruction field extractors
    function [4:0] get_rd;
        input [31:0] instruction;
        get_rd = instruction[11:7];
    endfunction

    function [4:0] get_rs1;
        input [31:0] instruction;
        get_rs1 = instruction[19:15];
    endfunction

    function [4:0] get_rs2;
        input [31:0] instruction;
        get_rs2 = instruction[24:20];
    endfunction

    function [31:0] get_imm_i;
        input [31:0] instruction;
        get_imm_i = {{20{instruction[31]}}, instruction[31:20]};
    endfunction

    function [31:0] get_imm_s;
        input [31:0] instruction;
        get_imm_s = {{20{instruction[31]}}, instruction[31:25], instruction[11:7]};
    endfunction

    function [31:0] get_imm_b;
        input [31:0] instruction;
        get_imm_b = {{20{instruction[31]}}, instruction[7], instruction[30:25], instruction[11:8], 1'b0};
    endfunction

    function [31:0] get_imm_j;
        input [31:0] instruction;
        get_imm_j = {{12{instruction[31]}}, instruction[19:12], instruction[20], instruction[30:21], 1'b0};
    endfunction

    function [31:0] get_imm_u;
        input [31:0] instruction;
        get_imm_u = {instruction[31:12], 12'h0};
    endfunction

    // Output assignments
    assign mem_data_out = memory[mem_addr[11:2]];
    assign mem_ready = 1'b1;
    assign execution_done = execution_done_reg;
    assign execution_error = execution_error_reg;
    assign user_cycles = user_cycles_reg;
    assign total_cycles = total_cycles_reg;
    assign current_pc = pc;
    assign segment_ready = segment_ready_reg;
    assign segment_data = segment_data_reg;

    // Memory write
    always @(posedge clk) begin
        if (mem_we) begin
            memory[mem_addr[11:2]] <= mem_data_in;
        end
    end

endmodule

// Simple testbench
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

    // Instantiate the module
    risc0_fpga_simple dut (
        .clk(clk),
        .rst_n(rst_n),
        .mem_addr(test_addr),
        .mem_data_in(test_data_in),
        .mem_data_out(test_data_out),
        .mem_we(test_we),
        .mem_re(test_re),
        .mem_ready(test_ready),
        .start_execution(start_execution),
        .segment_threshold(segment_threshold),
        .max_cycles(max_cycles),
        .execution_done(execution_done),
        .execution_error(execution_error),
        .user_cycles(user_cycles),
        .total_cycles(total_cycles),
        .current_pc(current_pc),
        .segment_ready(),
        .segment_data(),
        .segment_ack(1'b0)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
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
        max_cycles = 1000;  // Reduced for faster simulation

        $display("Starting simulation at time %t", $time);

        // Reset
        #100;
        rst_n = 1;
        #50;

        $display("Loading test program at time %t", $time);

        // Load test program
        write_memory(32'h0, 32'h00100093);  // addi x1, x0, 1
        write_memory(32'h4, 32'h00200113);  // addi x2, x0, 2
        write_memory(32'h8, 32'h00208133);  // add x2, x1, x2
        write_memory(32'hc, 32'h00000073);  // ecall

        $display("Starting execution at time %t", $time);

        // Start execution - keep it high for longer
        start_execution = 1;
        #100;  // Keep high for 100ns
        start_execution = 0;

        // Wait for completion with timeout
        #10000;  // 10us timeout

        // Display results
        $display("Execution completed:");
        $display("User cycles: %d", user_cycles);
        $display("Total cycles: %d", total_cycles);
        $display("Final PC: 0x%h", current_pc);
        $display("Execution done: %b", execution_done);
        $display("Execution error: %b", execution_error);

        if (execution_error)
            $display("ERROR: Execution failed");
        else if (execution_done)
            $display("SUCCESS: Execution completed successfully");
        else
            $display("TIMEOUT: Execution did not complete");

        #100;
        $finish;
    end

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
            $display("Wrote 0x%h to address 0x%h", data, addr);
        end
    endtask

    // Monitor execution
    always @(posedge clk) begin
        if (execution_done || execution_error) begin
            $display("Simulation finished at time %t", $time);
        end
    end

endmodule
