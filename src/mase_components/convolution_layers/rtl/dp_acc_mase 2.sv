`timescale 1ns / 1ps
module dp_acc_mase #(
    parameter DATA_IN_0_PRECISION_0 = 16,
    parameter WEIGHT_PRECISION_0 = 8,
    parameter DP_SIZE = 4,
    parameter ACC_DEPTH = 4,
    parameter DATA_OUT_0_PRECISION_0 = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
        DP_SIZE * ACC_DEPTH
    )
) (
    input clk,
    input rst,

    input  logic [DATA_IN_0_PRECISION_0 - 1:0] data_in_0      [DP_SIZE - 1 : 0],
    input                                      data_in_0_valid,
    output                                     data_in_0_ready,

    input  logic [WEIGHT_PRECISION_0 - 1:0] weight      [DP_SIZE - 1:0],
    input                                   weight_valid,
    output                                  weight_ready,

    output [DATA_OUT_0_PRECISION_0 - 1:0] data_out_0,
    output                                data_out_0_valid,
    input                                 data_out_0_ready
);
  localparam DP_WIDTH = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(DP_SIZE);
  logic [DP_WIDTH - 1:0] middle_result;
  logic middle_result_valid, middle_result_ready;

  fixed_dot_product #(
      .IN_WIDTH    (DATA_IN_0_PRECISION_0),
      .IN_SIZE     (DP_SIZE),
      .WEIGHT_WIDTH(WEIGHT_PRECISION_0)
  ) dot_product_inst (
      .clk           (clk),
      .rst           (rst),
      .data_in       (data_in_0),
      .data_in_valid (data_in_0_valid),
      .data_in_ready (data_in_0_ready),
      .weight        (weight),
      .weight_valid  (weight_valid),
      .weight_ready  (weight_ready),
      .data_out      (middle_result),
      .data_out_valid(middle_result_valid),
      .data_out_ready(middle_result_ready)
  );

  fixed_accumulator #(
      .IN_DEPTH(ACC_DEPTH),
      .IN_WIDTH(DP_WIDTH)
  ) acc_inst (
      .clk           (clk),
      .rst           (rst),
      .data_in       (middle_result),
      .data_in_valid (middle_result_valid),
      .data_in_ready (middle_result_ready),
      .data_out      (data_out_0),
      .data_out_valid(data_out_0_valid),
      .data_out_ready(data_out_0_ready)
  );
endmodule
