// ============================================================================
// SIMPLIFIED RISC-V PROCESSOR FOR BASYS 3 FPGA
// ============================================================================
// This is a simplified version optimized for the Basys 3 Artix-7 FPGA
// Removes complex instructions to fit within resource constraints

module risc0_fpga_executor (
    input wire clk,
    input wire rst_n,
    input wire start_execution,
    input wire [31:0] max_cycles,
    input wire [31:0] segment_threshold,
    input wire segment_ack,

    // Memory interface
    input wire [31:0] mem_addr,
    input wire [31:0] mem_data_in,
    input wire mem_we,
    output wire [31:0] mem_data_out,
    output wire mem_ready,

    // Status outputs
    output wire execution_done,
    output wire execution_error,
    output wire [63:0] user_cycles,
    output wire [63:0] total_cycles,
    output wire [31:0] current_pc,
    output wire [31:0] machine_mode,

    // Segment interface
    output wire segment_ready,
    output wire [31:0] segment_data
);

    // ============================================================================
    // REGISTER AND SIGNAL DECLARATIONS
    // ============================================================================

    // Execution state
    reg [2:0] exec_state;
    reg [31:0] pc;
    reg [31:0] user_pc;
    reg [31:0] machine_mode_reg;
    reg [63:0] user_cycles_reg;
    reg [63:0] total_cycles_reg;
    reg [31:0] segment_counter;

    // Memory and registers (reduced size)
    reg [31:0] registers [0:31];  // RISC-V registers
    reg [31:0] memory [0:511];    // Reduced memory (2KB instead of 4KB)

    // Instruction execution
    reg [31:0] current_instruction;
    reg [7:0] instruction_type;
    reg [31:0] decoded_instruction;

    // Segment management
    reg [31:0] segment_cycles;
    reg segment_ready_reg;
    reg [31:0] segment_data_reg;

    // Error handling
    reg execution_error_reg;
    reg execution_done_reg;

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
    localparam SEGMENT_CHECK = 3'b110;
    localparam ERROR = 3'b111;

    // ============================================================================
    // SIMPLIFIED INSTRUCTION TYPE DEFINITIONS
    // ============================================================================

    localparam INSN_ADD = 8'h00;
    localparam INSN_SUB = 8'h01;
    localparam INSN_XOR = 8'h02;
    localparam INSN_OR = 8'h03;
    localparam INSN_AND = 8'h04;
    localparam INSN_ADDI = 8'h07;
    localparam INSN_LUI = 8'h15;
    localparam INSN_LW = 8'h2A;
    localparam INSN_SW = 8'h32;
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

            // Initialize registers
            for (i = 0; i < 32; i = i + 1) begin
                registers[i] <= 32'h0;
            end

            // Memory initialization will be handled in the memory write block
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
                    end
                end

                FETCH: begin
                    // Fetch instruction from memory
                    current_instruction <= memory[pc[10:2]];  // Word-aligned access (reduced address bits)
                    exec_state <= DECODE;
                    total_cycles_reg <= total_cycles_reg + 1;
                end

                DECODE: begin
                    // Decode instruction
                    instruction_type <= decode_instruction_simple(current_instruction);
                    decoded_instruction <= current_instruction;
                    exec_state <= EXECUTE;
                end

                EXECUTE: begin
                    // Execute instruction
                    case (instruction_type)
                        INSN_ADD: begin
                            registers[get_rd(current_instruction)] <=
                                registers[get_rs1(current_instruction)] +
                                registers[get_rs2(current_instruction)];
                            pc <= pc + 4;
                        end

                        INSN_SUB: begin
                            registers[get_rd(current_instruction)] <=
                                registers[get_rs1(current_instruction)] -
                                registers[get_rs2(current_instruction)];
                            pc <= pc + 4;
                        end

                        INSN_XOR: begin
                            registers[get_rd(current_instruction)] <=
                                registers[get_rs1(current_instruction)] ^
                                registers[get_rs2(current_instruction)];
                            pc <= pc + 4;
                        end

                        INSN_OR: begin
                            registers[get_rd(current_instruction)] <=
                                registers[get_rs1(current_instruction)] |
                                registers[get_rs2(current_instruction)];
                            pc <= pc + 4;
                        end

                        INSN_AND: begin
                            registers[get_rd(current_instruction)] <=
                                registers[get_rs1(current_instruction)] &
                                registers[get_rs2(current_instruction)];
                            pc <= pc + 4;
                        end

                        INSN_ADDI: begin
                            registers[get_rd(current_instruction)] <=
                                registers[get_rs1(current_instruction)] +
                                get_imm_i(current_instruction);
                            pc <= pc + 4;
                        end

                        INSN_LUI: begin
                            registers[get_rd(current_instruction)] <=
                                get_imm_u(current_instruction);
                            pc <= pc + 4;
                        end

                        INSN_LW: begin
                            exec_state <= MEMORY;
                        end

                        INSN_SW: begin
                            exec_state <= MEMORY;
                        end

                        default: begin
                            exec_state <= ERROR;
                        end
                    endcase

                    user_cycles_reg <= user_cycles_reg + 1;
                    segment_cycles <= segment_cycles + 1;
                end

                MEMORY: begin
                    case (instruction_type)
                        INSN_LW: begin
                            registers[get_rd(current_instruction)] <=
                                memory[get_rs1(current_instruction) + get_imm_i(current_instruction)];
                        end

                        INSN_SW: begin
                            // Memory write will be handled in the memory write block
                            mem_write_addr <= get_rs1(current_instruction) + get_imm_s(current_instruction);
                            mem_write_data <= registers[get_rs2(current_instruction)];
                            mem_write_enable <= 1'b1;
                        end
                    endcase

                    pc <= pc + 4;
                    exec_state <= SEGMENT_CHECK;
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
    // SIMPLIFIED INSTRUCTION DECODER FUNCTION
    // ============================================================================

    function [7:0] decode_instruction_simple;
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
                    10'h000: decode_instruction_simple = INSN_ADD;
                    10'h020: decode_instruction_simple = INSN_SUB;
                    10'h400: decode_instruction_simple = INSN_XOR;
                    10'h600: decode_instruction_simple = INSN_OR;
                    10'h700: decode_instruction_simple = INSN_AND;
                    default: decode_instruction_simple = INSN_INVALID;
                endcase
            end

            7'h13: begin  // I-format arithmetic
                case (func3)
                    3'h0: decode_instruction_simple = INSN_ADDI;
                    default: decode_instruction_simple = INSN_INVALID;
                endcase
            end

            7'h37: begin  // U-format
                decode_instruction_simple = INSN_LUI;
            end

            7'h03: begin  // Load instructions
                case (func3)
                    3'h2: decode_instruction_simple = INSN_LW;
                    default: decode_instruction_simple = INSN_INVALID;
                endcase
            end

            7'h23: begin  // Store instructions
                case (func3)
                    3'h2: decode_instruction_simple = INSN_SW;
                    default: decode_instruction_simple = INSN_INVALID;
                endcase
            end

            default: decode_instruction_simple = INSN_INVALID;
        endcase
    end
    endfunction

    // ============================================================================
    // INSTRUCTION FIELD EXTRACTION FUNCTIONS
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

    function [31:0] get_imm_u;
        input [31:0] instruction;
        get_imm_u = {instruction[31:12], 12'h0};
    endfunction

    // ============================================================================
    // OUTPUT ASSIGNMENTS
    // ============================================================================

    assign mem_data_out = memory[mem_addr[10:2]];  // Reduced address bits
    assign mem_ready = 1'b1;  // Always ready for simplicity

    assign execution_done = execution_done_reg;
    assign execution_error = execution_error_reg;
    assign user_cycles = user_cycles_reg;
    assign total_cycles = total_cycles_reg;
    assign current_pc = pc;
    assign machine_mode = machine_mode_reg;

    assign segment_ready = segment_ready_reg;
    assign segment_data = segment_data_reg;

    // ============================================================================
    // MEMORY WRITE AND INITIALIZATION
    // ============================================================================

    always @(posedge clk) begin
        if (!rst_n) begin
            // Initialize memory to zero
            for (i = 0; i < 512; i = i + 1) begin
                memory[i] <= 32'h0;
            end
        end else if (mem_we) begin
            // Write from external interface
            memory[mem_addr[10:2]] <= mem_data_in;  // Reduced address bits
        end else if (mem_write_enable) begin
            // Write from SW instruction
            memory[mem_write_addr[10:2]] <= mem_write_data;  // Reduced address bits
        end
    end

endmodule
