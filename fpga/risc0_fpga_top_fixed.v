// RISC0 FPGA Top-Level Module - Fixed Implementation
// Integrates RISC-V executor with host interface and status outputs
// Copyright 2025 RISC Zero, Inc.

`timescale 1ns / 1ps

module risc0_fpga_top (
    input wire clk,
    input wire rst_n,

    // Host interface
    input wire [31:0] host_addr,
    input wire [31:0] host_data_in,
    output wire [31:0] host_data_out,
    input wire host_we,
    input wire host_re,
    output wire host_ready,

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

    // UART interface
    output wire uart_tx,
    input wire uart_rx,

    // LED status outputs
    output wire [15:0] led,

    // Switch inputs
    input wire [7:0] sw,

    // Button inputs
    input wire [3:0] btn
);

    // ============================================================================
    // INTERNAL SIGNALS
    // ============================================================================

    // Memory interface signals
    wire [31:0] mem_addr, mem_data_in, mem_data_out;
    wire mem_we, mem_re, mem_ready;

    // Machine mode and segment interface
    wire [31:0] machine_mode;
    wire segment_ready, segment_ack;
    wire [31:0] segment_data;

    // UART communication signals
    wire uart_tx_enable;
    wire [7:0] uart_tx_data;
    wire uart_tx_busy;
    wire uart_rx_valid;
    wire [7:0] uart_rx_data;

    // Status and control signals
    wire execution_active;
    wire memory_access;
    wire error_state;
    wire debug_mode;

    // ============================================================================
    // RISC-V EXECUTOR INSTANCE
    // ============================================================================

    risc0_fpga_executor executor (
        .clk(clk),
        .rst_n(rst_n),
        .mem_addr(mem_addr),
        .mem_data_in(mem_data_in),
        .mem_data_out(mem_data_out),
        .mem_we(mem_we),
        .mem_re(mem_re),
        .mem_ready(mem_ready),
        .start_execution(start_execution),
        .segment_threshold(segment_threshold),
        .max_cycles(max_cycles),
        .execution_done(execution_done),
        .execution_error(execution_error),
        .user_cycles(user_cycles),
        .total_cycles(total_cycles),
        .current_pc(current_pc),
        .machine_mode(machine_mode),
        .segment_ready(segment_ready),
        .segment_data(segment_data),
        .segment_ack(segment_ack)
    );

    // ============================================================================
    // HOST INTERFACE LOGIC
    // ============================================================================

    // Connect host interface to memory interface
    assign host_data_out = mem_data_out;
    assign host_ready = mem_ready;
    assign mem_addr = host_addr;
    assign mem_data_in = host_data_in;
    assign mem_we = host_we;
    assign mem_re = host_re;

    // ============================================================================
    // UART COMMUNICATION MODULE
    // ============================================================================

    uart_controller uart_ctrl (
        .clk(clk),
        .rst_n(rst_n),
        .uart_tx(uart_tx),
        .uart_rx(uart_rx),
        .tx_enable(uart_tx_enable),
        .tx_data(uart_tx_data),
        .tx_busy(uart_tx_busy),
        .rx_valid(uart_rx_valid),
        .rx_data(uart_rx_data)
    );

    // ============================================================================
    // CLOCK DOMAIN CROSSING FOR UART
    // ============================================================================

    // UART clock domain crossing registers
    reg [7:0] uart_tx_data_sync;
    reg uart_tx_enable_sync;
    reg [7:0] uart_rx_data_sync;
    reg uart_rx_valid_sync;

    // CDC for UART TX
    always @(posedge clk) begin
        uart_tx_data_sync <= uart_tx_data;
        uart_tx_enable_sync <= uart_tx_enable;
    end

    // CDC for UART RX
    always @(posedge clk) begin
        uart_rx_data_sync <= uart_rx_data;
        uart_rx_valid_sync <= uart_rx_valid;
    end

    // ============================================================================
    // STATUS LED CONTROL
    // ============================================================================

    // LED assignments for status indication
    assign led[0] = execution_active;      // Execution active
    assign led[1] = memory_access;         // Memory access
    assign led[2] = error_state;           // Error state
    assign led[3] = segment_ready;         // Segment ready
    assign led[4] = current_pc[0];         // PC bit 0
    assign led[5] = current_pc[1];         // PC bit 1
    assign led[6] = current_pc[2];         // PC bit 2
    assign led[7] = current_pc[3];         // PC bit 3
    assign led[8] = user_cycles[0];        // User cycles bit 0
    assign led[9] = user_cycles[1];        // User cycles bit 1
    assign led[10] = user_cycles[2];       // User cycles bit 2
    assign led[11] = user_cycles[3];       // User cycles bit 3
    assign led[12] = execution_done;       // Execution done
    assign led[13] = execution_error;      // Execution error
    assign led[14] = debug_mode;           // Debug mode
    assign led[15] = machine_mode[0];      // Machine mode bit 0

    // ============================================================================
    // STATUS SIGNAL GENERATION
    // ============================================================================

    // Execution active signal
    assign execution_active = (current_pc != 32'h0) && !execution_done && !execution_error;

    // Memory access signal (simplified)
    assign memory_access = mem_we || mem_re;

    // Error state signal
    assign error_state = execution_error;

    // Debug mode from switches
    assign debug_mode = sw[0];

    // ============================================================================
    // SEGMENT ACKNOWLEDGMENT LOGIC
    // ============================================================================

    // Simple segment acknowledgment (can be enhanced with UART)
    assign segment_ack = segment_ready;

    // ============================================================================
    // UART TRANSMISSION LOGIC
    // ============================================================================

    // Send status information via UART when requested
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            uart_tx_enable <= 1'b0;
            uart_tx_data <= 8'h00;
        end else begin
            // Send status when execution completes
            if (execution_done && !uart_tx_busy) begin
                uart_tx_enable <= 1'b1;
                uart_tx_data <= 8'hAA; // Status marker
            end else begin
                uart_tx_enable <= 1'b0;
            end
        end
    end

    // ============================================================================
    // BUTTON CONTROL LOGIC
    // ============================================================================

    // Button debouncing and control
    reg [3:0] btn_prev;
    reg [3:0] btn_pressed;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            btn_prev <= 4'h0;
            btn_pressed <= 4'h0;
        end else begin
            btn_prev <= btn;
            btn_pressed <= btn & ~btn_prev; // Rising edge detection
        end
    end

    // ============================================================================
    // SWITCH CONFIGURATION LOGIC
    // ============================================================================

    // Switch-based configuration
    reg [31:0] config_segment_threshold;
    reg [31:0] config_max_cycles;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            config_segment_threshold <= 32'd1000;
            config_max_cycles <= 32'd10000;
        end else begin
            // Configure based on switches
            case (sw[7:4])
                4'h0: config_segment_threshold <= 32'd100;
                4'h1: config_segment_threshold <= 32'd500;
                4'h2: config_segment_threshold <= 32'd1000;
                4'h3: config_segment_threshold <= 32'd2000;
                default: config_segment_threshold <= 32'd1000;
            endcase

            case (sw[3:0])
                4'h0: config_max_cycles <= 32'd1000;
                4'h1: config_max_cycles <= 32'd5000;
                4'h2: config_max_cycles <= 32'd10000;
                4'h3: config_max_cycles <= 32'd50000;
                default: config_max_cycles <= 32'd10000;
            endcase
        end
    end

    // Use configured values if not provided externally
    wire [31:0] effective_segment_threshold = (segment_threshold == 32'h0) ?
                                             config_segment_threshold : segment_threshold;
    wire [31:0] effective_max_cycles = (max_cycles == 32'h0) ?
                                       config_max_cycles : max_cycles;

    // ============================================================================
    // DEBUG AND MONITORING
    // ============================================================================

    // Debug output via UART when debug mode is enabled
    always @(posedge clk) begin
        if (debug_mode && execution_done) begin
            $display("FPGA Execution Complete:");
            $display("  PC: 0x%h", current_pc);
            $display("  User Cycles: %d", user_cycles);
            $display("  Total Cycles: %d", total_cycles);
            $display("  Machine Mode: %d", machine_mode);
            if (execution_error)
                $display("  ERROR: Execution failed");
        end
    end

endmodule

// ============================================================================
// UART CONTROLLER MODULE
// ============================================================================

module uart_controller (
    input wire clk,
    input wire rst_n,
    output reg uart_tx,
    input wire uart_rx,
    input wire tx_enable,
    input wire [7:0] tx_data,
    output reg tx_busy,
    output reg rx_valid,
    output reg [7:0] rx_data
);

    // UART parameters (115200 baud at 100MHz)
    localparam BAUD_RATE = 868; // 100MHz / 115200

    // TX state machine
    reg [2:0] tx_state;
    reg [7:0] tx_shift;
    reg [9:0] tx_counter;
    reg [3:0] tx_bit_count;

    // RX state machine
    reg [2:0] rx_state;
    reg [7:0] rx_shift;
    reg [9:0] rx_counter;
    reg [3:0] rx_bit_count;

    // TX state machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx_state <= 3'h0;
            uart_tx <= 1'b1;
            tx_busy <= 1'b0;
            tx_counter <= 10'h0;
            tx_bit_count <= 4'h0;
        end else begin
            case (tx_state)
                3'h0: begin // Idle
                    uart_tx <= 1'b1;
                    if (tx_enable) begin
                        tx_state <= 3'h1;
                        tx_shift <= tx_data;
                        tx_busy <= 1'b1;
                        tx_counter <= 10'h0;
                        tx_bit_count <= 4'h0;
                    end
                end

                3'h1: begin // Start bit
                    uart_tx <= 1'b0;
                    if (tx_counter >= BAUD_RATE - 1) begin
                        tx_state <= 3'h2;
                        tx_counter <= 10'h0;
                    end else begin
                        tx_counter <= tx_counter + 1;
                    end
                end

                3'h2: begin // Data bits
                    uart_tx <= tx_shift[0];
                    if (tx_counter >= BAUD_RATE - 1) begin
                        tx_shift <= {1'b0, tx_shift[7:1]};
                        tx_counter <= 10'h0;
                        if (tx_bit_count >= 7) begin
                            tx_state <= 3'h3;
                        end else begin
                            tx_bit_count <= tx_bit_count + 1;
                        end
                    end else begin
                        tx_counter <= tx_counter + 1;
                    end
                end

                3'h3: begin // Stop bit
                    uart_tx <= 1'b1;
                    if (tx_counter >= BAUD_RATE - 1) begin
                        tx_state <= 3'h0;
                        tx_busy <= 1'b0;
                    end else begin
                        tx_counter <= tx_counter + 1;
                    end
                end

                default: tx_state <= 3'h0;
            endcase
        end
    end

    // RX state machine (simplified)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_state <= 3'h0;
            rx_valid <= 1'b0;
            rx_data <= 8'h00;
            rx_counter <= 10'h0;
            rx_bit_count <= 4'h0;
        end else begin
            rx_valid <= 1'b0;
            case (rx_state)
                3'h0: begin // Idle, waiting for start bit
                    if (!uart_rx) begin // Start bit detected
                        rx_state <= 3'h1;
                        rx_counter <= 10'h0;
                        rx_bit_count <= 4'h0;
                    end
                end

                3'h1: begin // Sample data bits
                    if (rx_counter >= BAUD_RATE/2) begin
                        rx_shift[rx_bit_count] <= uart_rx;
                        rx_counter <= 10'h0;
                        if (rx_bit_count >= 7) begin
                            rx_state <= 3'h2;
                        end else begin
                            rx_bit_count <= rx_bit_count + 1;
                        end
                    end else begin
                        rx_counter <= rx_counter + 1;
                    end
                end

                3'h2: begin // Stop bit
                    if (rx_counter >= BAUD_RATE - 1) begin
                        rx_state <= 3'h0;
                        rx_data <= rx_shift;
                        rx_valid <= 1'b1;
                    end else begin
                        rx_counter <= rx_counter + 1;
                    end
                end

                default: rx_state <= 3'h0;
            endcase
        end
    end

endmodule

