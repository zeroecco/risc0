// RISC0 FPGA RISC-V Executor - Fixed Implementation
// Copyright 2025 RISC Zero, Inc.

`timescale 1ns / 1ps

module risc0_fpga_executor (
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
    output wire [31:0] machine_mode,

    // Segment interface
    output wire segment_ready,
    output wire [31:0] segment_data,
    input wire segment_ack
);

    // ============================================================================
    // INTERNAL SIGNALS AND REGISTERS
    // ============================================================================

    // Execution state
    reg [2:0] exec_state;
    reg [31:0] pc;
    reg [31:0] user_pc;
    reg [31:0] machine_mode_reg;
    reg [63:0] user_cycles_reg;
    reg [63:0] total_cycles_reg;
    reg [31:0] segment_counter;

    // Memory and registers - Use BRAM for synthesis
    (* ram_style = "block" *)
    reg [31:0] registers [0:31];  // RISC-V registers
    (* ram_style = "block" *)
    reg [31:0] memory [0:4095];   // 16KB memory (4096 words)

    // Instruction execution
    reg [31:0] current_instruction;
    reg [7:0] instruction_type;
    reg [31:0] decoded_instruction;
    reg [31:0] alu_result;
    reg [31:0] mem_address;
    reg mem_access_pending;

    // Segment management
    reg [31:0] segment_cycles;
    reg segment_ready_reg;
    reg [31:0] segment_data_reg;

    // Error handling
    reg execution_error_reg;
    reg execution_done_reg;
    reg [31:0] error_code;

    // Memory write control signals
    reg [31:0] mem_write_addr;
    reg [31:0] mem_write_data;
    reg mem_write_enable;

    // Loop variables for initialization
    integer i;

    // ============================================================================
    // STATE MACHINE DEFINITIONS
    // ============================================================================

    localparam IDLE = 3'b000;
    localparam FETCH = 3'b001;
    localparam DECODE = 3'b010;
    localparam EXECUTE = 3'b011;
    localparam MEMORY = 3'b100;
    localparam WRITEBACK = 3'b101;
    localparam SEGMENT_CHECK = 3'b110;
    localparam ERROR = 3'b111;

    // ============================================================================
    // INSTRUCTION TYPE DEFINITIONS
    // ============================================================================

    localparam INSN_ADD = 8'h00;
    localparam INSN_SUB = 8'h01;
    localparam INSN_XOR = 8'h02;
    localparam INSN_OR = 8'h03;
    localparam INSN_AND = 8'h04;
    localparam INSN_SLT = 8'h05;
    localparam INSN_SLTU = 8'h06;
    localparam INSN_ADDI = 8'h07;
    localparam INSN_XORI = 8'h08;
    localparam INSN_ORI = 8'h09;
    localparam INSN_ANDI = 8'h0A;
    localparam INSN_SLTI = 8'h0B;
    localparam INSN_SLTIU = 8'h0C;
    localparam INSN_BEQ = 8'h0D;
    localparam INSN_BNE = 8'h0E;
    localparam INSN_BLT = 8'h0F;
    localparam INSN_BGE = 8'h10;
    localparam INSN_BLTU = 8'h11;
    localparam INSN_BGEU = 8'h12;
    localparam INSN_JAL = 8'h13;
    localparam INSN_JALR = 8'h14;
    localparam INSN_LUI = 8'h15;
    localparam INSN_AUIPC = 8'h16;
    localparam INSN_SLL = 8'h18;
    localparam INSN_SLLI = 8'h19;
    localparam INSN_SRL = 8'h20;
    localparam INSN_SRA = 8'h21;
    localparam INSN_SRLI = 8'h22;
    localparam INSN_SRAI = 8'h23;
    localparam INSN_LB = 8'h28;
    localparam INSN_LH = 8'h29;
    localparam INSN_LW = 8'h2A;
    localparam INSN_LBU = 8'h2B;
    localparam INSN_LHU = 8'h2C;
    localparam INSN_SB = 8'h30;
    localparam INSN_SH = 8'h31;
    localparam INSN_SW = 8'h32;
    localparam INSN_EANY = 8'h38;
    localparam INSN_MRET = 8'h39;
    localparam INSN_INVALID = 8'hFF;

    // ============================================================================
    // MAIN EXECUTION STATE MACHINE
    // ============================================================================

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            exec_state <= IDLE;
            pc <= 32'h0;
            user_pc <= 32'h0;
            machine_mode_reg <= 32'h0;
            user_cycles_reg <= 64'h0;
            total_cycles_reg <= 64'h0;
            segment_counter <= 32'h0;
            segment_cycles <= 32'h0;
            segment_ready_reg <= 1'b0;
            segment_data_reg <= 32'h0;
            execution_error_reg <= 1'b0;
            execution_done_reg <= 1'b0;
            mem_write_enable <= 1'b0;
            mem_access_pending <= 1'b0;
            error_code <= 32'h0;

            // Initialize registers
            for (i = 0; i < 32; i = i + 1) begin
                registers[i] <= 32'h0;
            end

            // Initialize memory to zero
            for (i = 0; i < 4096; i = i + 1) begin
                memory[i] <= 32'h0;
            end
        end else begin
            case (exec_state)
                IDLE: begin
                    if (start_execution) begin
                        exec_state <= FETCH;
                        pc <= 32'h0;
                        user_cycles_reg <= 64'h0;
                        total_cycles_reg <= 64'h0;
                        segment_counter <= 32'h0;
                        segment_cycles <= 32'h0;
                        execution_error_reg <= 1'b0;
                        execution_done_reg <= 1'b0;
                        error_code <= 32'h0;
                    end
                end

                FETCH: begin
                    // Bounds check for memory access
                    if (pc[31:12] != 20'h0) begin
                        // Address out of range
                        error_code <= 32'h1;
                        exec_state <= ERROR;
                    end else begin
                        // Fetch instruction from memory
                        current_instruction <= memory[pc[13:2]];  // 16KB aligned access
                        exec_state <= DECODE;
                        total_cycles_reg <= total_cycles_reg + 1;
                    end
                end

                DECODE: begin
                    // Decode instruction
                    instruction_type <= decode_instruction(current_instruction);
                    decoded_instruction <= current_instruction;
                    exec_state <= EXECUTE;
                end

                EXECUTE: begin
                    // Execute instruction
                    case (instruction_type)
                        INSN_ADD: begin
                            alu_result <= registers[get_rs1(current_instruction)] +
                                         registers[get_rs2(current_instruction)];
                            if (get_rd(current_instruction) != 0) begin
                                registers[get_rd(current_instruction)] <= alu_result;
                            end
                            pc <= pc + 4;
                            exec_state <= SEGMENT_CHECK;
                        end

                        INSN_SUB: begin
                            alu_result <= registers[get_rs1(current_instruction)] -
                                         registers[get_rs2(current_instruction)];
                            if (get_rd(current_instruction) != 0) begin
                                registers[get_rd(current_instruction)] <= alu_result;
                            end
                            pc <= pc + 4;
                            exec_state <= SEGMENT_CHECK;
                        end

                        INSN_ADDI: begin
                            alu_result <= registers[get_rs1(current_instruction)] +
                                         get_imm_i(current_instruction);
                            if (get_rd(current_instruction) != 0) begin
                                registers[get_rd(current_instruction)] <= alu_result;
                            end
                            pc <= pc + 4;
                            exec_state <= SEGMENT_CHECK;
                        end

                        INSN_LUI: begin
                            if (get_rd(current_instruction) != 0) begin
                                registers[get_rd(current_instruction)] <= get_imm_u(current_instruction);
                            end
                            pc <= pc + 4;
                            exec_state <= SEGMENT_CHECK;
                        end

                        INSN_LW: begin
                            mem_address <= registers[get_rs1(current_instruction)] +
                                         get_imm_i(current_instruction);
                            mem_access_pending <= 1'b1;
                            exec_state <= MEMORY;
                        end

                        INSN_SW: begin
                            mem_address <= registers[get_rs1(current_instruction)] +
                                         get_imm_s(current_instruction);
                            mem_write_addr <= registers[get_rs1(current_instruction)] +
                                            get_imm_s(current_instruction);
                            mem_write_data <= registers[get_rs2(current_instruction)];
                            mem_write_enable <= 1'b1;
                            pc <= pc + 4;
                            exec_state <= SEGMENT_CHECK;
                        end

                        INSN_BEQ: begin
                            if (registers[get_rs1(current_instruction)] ==
                                registers[get_rs2(current_instruction)]) begin
                                pc <= pc + get_imm_b(current_instruction);
                            end else begin
                                pc <= pc + 4;
                            end
                            exec_state <= SEGMENT_CHECK;
                        end

                        INSN_JAL: begin
                            if (get_rd(current_instruction) != 0) begin
                                registers[get_rd(current_instruction)] <= pc + 4;
                            end
                            pc <= pc + get_imm_j(current_instruction);
                            exec_state <= SEGMENT_CHECK;
                        end

                        INSN_EANY: begin
                            // System call - handle in software
                            exec_state <= SEGMENT_CHECK;
                        end

                        default: begin
                            // Invalid instruction
                            error_code <= 32'h2;
                            exec_state <= ERROR;
                        end
                    endcase

                    user_cycles_reg <= user_cycles_reg + 1;
                    segment_cycles <= segment_cycles + 1;
                end

                MEMORY: begin
                    case (instruction_type)
                        INSN_LW: begin
                            // Bounds check for memory access
                            if (mem_address[31:14] != 18'h0) begin
                                error_code <= 32'h3;
                                exec_state <= ERROR;
                            end else begin
                                if (get_rd(current_instruction) != 0) begin
                                    registers[get_rd(current_instruction)] <= memory[mem_address[13:2]];
                                end
                                pc <= pc + 4;
                                mem_access_pending <= 1'b0;
                                exec_state <= SEGMENT_CHECK;
                            end
                        end
                    endcase
                end

                SEGMENT_CHECK: begin
                    // Clear memory write enable after SW instruction
                    mem_write_enable <= 1'b0;

                    if (segment_cycles >= segment_threshold) begin
                        // Create segment
                        segment_ready_reg <= 1'b1;
                        segment_data_reg <= segment_counter;
                        segment_counter <= segment_counter + 1;
                        segment_cycles <= 32'h0;

                        if (segment_ack) begin
                            segment_ready_reg <= 1'b0;
                            exec_state <= FETCH;
                        end
                    end else if (total_cycles_reg >= max_cycles) begin
                        execution_done_reg <= 1'b1;
                        exec_state <= IDLE;
                    end else begin
                        exec_state <= FETCH;
                    end
                end

                ERROR: begin
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

    // ============================================================================
    // INSTRUCTION DECODER FUNCTION
    // ============================================================================

    function [7:0] decode_instruction;
        input [31:0] instruction;
        reg [6:0] opcode;
        reg [2:0] func3;
        reg [6:0] func7;
    begin
        opcode = instruction[6:0];
        func3 = instruction[14:12];
        func7 = instruction[31:25];

        case (opcode)
            7'h33: begin  // R-format arithmetic
                case ({func3, func7})
                    10'h000: decode_instruction = INSN_ADD;
                    10'h020: decode_instruction = INSN_SUB;
                    10'h100: decode_instruction = INSN_SLL;
                    10'h200: decode_instruction = INSN_SLT;
                    10'h300: decode_instruction = INSN_SLTU;
                    10'h400: decode_instruction = INSN_XOR;
                    10'h500: decode_instruction = INSN_SRL;
                    10'h520: decode_instruction = INSN_SRA;
                    10'h600: decode_instruction = INSN_OR;
                    10'h700: decode_instruction = INSN_AND;
                    default: decode_instruction = INSN_INVALID;
                endcase
            end

            7'h13: begin  // I-format arithmetic
                case (func3)
                    3'h0: decode_instruction = INSN_ADDI;
                    3'h1: decode_instruction = (func7 == 7'h00) ? INSN_SLLI : INSN_INVALID;
                    3'h2: decode_instruction = INSN_SLTI;
                    3'h3: decode_instruction = INSN_SLTIU;
                    3'h4: decode_instruction = INSN_XORI;
                    3'h5: decode_instruction = (func7 == 7'h00) ? INSN_SRLI :
                                 (func7 == 7'h20) ? INSN_SRAI : INSN_INVALID;
                    3'h6: decode_instruction = INSN_ORI;
                    3'h7: decode_instruction = INSN_ANDI;
                    default: decode_instruction = INSN_INVALID;
                endcase
            end

            7'h03: begin  // I-format loads
                case (func3)
                    3'h0: decode_instruction = INSN_LB;
                    3'h1: decode_instruction = INSN_LH;
                    3'h2: decode_instruction = INSN_LW;
                    3'h4: decode_instruction = INSN_LBU;
                    3'h5: decode_instruction = INSN_LHU;
                    default: decode_instruction = INSN_INVALID;
                endcase
            end

            7'h23: begin  // S-format stores
                case (func3)
                    3'h0: decode_instruction = INSN_SB;
                    3'h1: decode_instruction = INSN_SH;
                    3'h2: decode_instruction = INSN_SW;
                    default: decode_instruction = INSN_INVALID;
                endcase
            end

            7'h37: decode_instruction = INSN_LUI;    // U-format lui
            7'h17: decode_instruction = INSN_AUIPC;  // U-format auipc

            7'h63: begin  // B-format branches
                case (func3)
                    3'h0: decode_instruction = INSN_BEQ;
                    3'h1: decode_instruction = INSN_BNE;
                    3'h4: decode_instruction = INSN_BLT;
                    3'h5: decode_instruction = INSN_BGE;
                    3'h6: decode_instruction = INSN_BLTU;
                    3'h7: decode_instruction = INSN_BGEU;
                    default: decode_instruction = INSN_INVALID;
                endcase
            end

            7'h6F: decode_instruction = INSN_JAL;   // J-format jal
            7'h67: decode_instruction = INSN_JALR;  // I-format jalr

            7'h73: begin  // System instructions
                case ({func3, func7})
                    10'h018: decode_instruction = INSN_MRET;
                    10'h000: decode_instruction = INSN_EANY;
                    default: decode_instruction = INSN_INVALID;
                endcase
            end

            default: decode_instruction = INSN_INVALID;
        endcase
    end
    endfunction

    // ============================================================================
    // INSTRUCTION FIELD EXTRACTORS
    // ============================================================================

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

    // ============================================================================
    // OUTPUT ASSIGNMENTS
    // ============================================================================

    assign mem_data_out = (mem_addr[31:14] == 18'h0) ? memory[mem_addr[13:2]] : 32'h0;
    assign mem_ready = 1'b1;  // Always ready for simplicity

    assign execution_done = execution_done_reg;
    assign execution_error = execution_error_reg;
    assign user_cycles = user_cycles_reg;
    assign total_cycles = total_cycles_reg;
    assign current_pc = pc;
    assign machine_mode = machine_mode_reg;

    assign segment_ready = segment_ready_reg;
    assign segment_data = segment_data_reg;

    // Memory write and initialization
    always @(posedge clk) begin
        if (!rst_n) begin
            // Memory initialization handled in main state machine
        end else if (mem_we && mem_addr[31:14] == 18'h0) begin
            // Write from external interface
            memory[mem_addr[13:2]] <= mem_data_in;
        end else if (mem_write_enable && mem_write_addr[31:14] == 18'h0) begin
            // Write from SW instruction
            memory[mem_write_addr[13:2]] <= mem_write_data;
        end
    end

endmodule


