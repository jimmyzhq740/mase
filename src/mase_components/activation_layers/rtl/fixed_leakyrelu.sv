`timescale 1ns / 1ps

module fixed_leakyrelu #(
    /* verilator lint_off UNUSEDPARAM */
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 8,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 1,

    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 3,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 8,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 1,

    parameter NEGATIVE_SLOPE = 2,
    parameter NEGATIVE_SLOPE_PRECISION_0 = 8, //since negative slope is normally less than 1, NEGATIVE_SLOPE_PRECISION_1 ahould have more bits
    parameter NEGATIVE_SLOPE_PRECISION_1 = 7,
    parameter INPLACE = 0
) (
    /* verilator lint_off UNUSEDSIGNAL */
    input rst,
    input clk,
    input logic [DATA_IN_0_PRECISION_0-1:0] data_in_0[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0[DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],

    input  logic data_in_0_valid,
    output logic data_in_0_ready,
    output logic data_out_0_valid,
    input  logic data_out_0_ready
);
  // To transfer to fixed point for Negative Slope Value, Example: Slope = 0.5 => 0.5 * 2^7 = 64 decimal => 8'h40
  localparam int NEG_SLOPE_QUAN = $rtoi(NEGATIVE_SLOPE * (1 << NEGATIVE_SLOPE_PRECISION_1));

  // The same assertion as other for data stability
  initial begin
    assert (DATA_IN_0_PRECISION_0 == DATA_OUT_0_PRECISION_0)
    else $error("LeakyReLU: DATA_IN_0_PRECISION_0 != DATA_OUT_0_PRECISION_0");
    assert (DATA_IN_0_PRECISION_1 == DATA_OUT_0_PRECISION_1)
    else $error("LeakyReLU: DATA_IN_0_PRECISION_1 != DATA_OUT_0_PRECISION_1");
  end

  /* verilator lint_off SELRANGE */
  for (
      genvar i = 0; i < DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1; i++
  ) begin : ReLU
    always_comb begin
      if ($signed(data_in_0[i]) < 0)
        //Multiply by the slope constant in the same fixed‐point domain:
        data_out_0[i] = $signed(
            data_in_0[i]
        ) * NEG_SLOPE_QUAN >>> NEGATIVE_SLOPE_PRECISION_1;

      else data_out_0[i] = data_in_0[i];
    end
  end

  assign data_out_0_valid = data_in_0_valid;
  assign data_in_0_ready  = data_out_0_ready;


endmodule
