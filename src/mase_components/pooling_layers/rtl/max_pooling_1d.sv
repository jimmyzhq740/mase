`timescale 1ns / 1ps

module max_pooling_1d #(
    // Data precision parameters (signed)
    parameter DATA_IN_0_PRECISION_0  = 8,
    parameter DATA_IN_0_PRECISION_1  = 5,
    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 5,

    // Pooling parameters (for a 4x4 input and 2x2 pooling)
    parameter POOL_SIZE                   = 2,
    parameter STRIDE                      = 0,
    parameter PADDING                     = 0,

    // Input interface: assumes 4 data per cycle (representing one full row)
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 2,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 2,

    // Unused parameters (for matching external interface with 4D tensor)
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_3 = 1,

    parameter DATA_IN_0_PARALLELISM_DIM_2  = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_3  = 1,

    // Output interface: assumes output of 4 data (corresponding to 4 pooling blocks)
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 4,

    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_2 = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_3 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_3 = 1,

    // FIFO related parameter (adjustable)
    parameter FIFO_DEPTH = 4
) (
    input logic clk,
    input logic rst,
    
    // Declare input data as signed
    input  logic signed [DATA_IN_0_PRECISION_0-1:0] data_in_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input logic data_in_0_valid,
    output logic data_in_0_ready,
    
    // Declare output data as signed
    output logic signed [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_0_valid,
    input logic data_out_0_ready
);
    localparam DATA_WIDTH = DATA_IN_0_PRECISION_0;

  logic signed [DATA_WIDTH-1:0] window_regs[0:POOL_SIZE-1];
  logic signed [DATA_WIDTH-1:0] max_value;
  logic [$clog2(POOL_SIZE+1)-1:0] element_count;

  typedef enum logic [1:0] {
    IDLE,
    BUFFER,
    PROCESS,
    OUTPUT
  } state_t;

  state_t current_state, next_state;

  always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
      current_state <= IDLE;
      element_count <= 0;
      data_out_0_valid <= 0;  // Fixed: Initialize inside reset
    end else begin
      current_state <= next_state;
    end
  end

  always_comb begin
    next_state = current_state;
    case (current_state)
      IDLE: if (data_in_0_valid) next_state = BUFFER;
      BUFFER: if (element_count == POOL_SIZE-1) next_state = PROCESS;
      PROCESS: next_state = OUTPUT;
      OUTPUT: if (data_out_0_ready) next_state = IDLE;
    endcase
  end

  always_ff @(posedge clk) begin
    if (!rst && data_in_0_valid) begin
      for (int i = 0; i < POOL_SIZE-1; i = i + 1) begin
        window_regs[i+1] <= window_regs[i];
      end
      window_regs[0] <= data_in_0;
    end
  end

  always_ff @(posedge clk) begin
    if (current_state == PROCESS) begin
      max_value = window_regs[0];
      for (int i = 1; i < POOL_SIZE; i = i + 1) begin
        if (window_regs[i] > max_value) max_value = window_regs[i];
      end
    end
  end

  always_ff @(posedge clk) begin
    if (current_state == OUTPUT) begin
      data_out_0 <= max_value;  // Fixed: No array context issue
    end
  end

  // Fixed: Ensure data_out_0_valid has a single driver
  always_ff @(posedge clk) begin
    if (rst) begin
      data_out_0_valid <= 0;
    end else if (current_state == OUTPUT) begin
      data_out_0_valid <= 1'b1;
    end else if (data_out_0_ready) begin
      data_out_0_valid <= 1'b0;
    end
  end

endmodule