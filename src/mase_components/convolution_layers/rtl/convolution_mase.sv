`timescale 1ns / 1ps
module convolution_mase #(
    //

    parameter DATA_IN_0_PRECISION_0       = 16,
    parameter DATA_IN_0_PRECISION_1       = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 3,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = 2,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_3 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_2 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_3 = 4,

    parameter WEIGHT_PRECISION_0       = 8,
    parameter WEIGHT_PRECISION_1       = 4,
    parameter WEIGHT_TENSOR_SIZE_DIM_0 = 3,
    parameter WEIGHT_TENSOR_SIZE_DIM_1 = 3,
    parameter WEIGHT_TENSOR_SIZE_DIM_2 = 1,
    parameter WEIGHT_TENSOR_SIZE_DIM_3 = 2,
    parameter WEIGHT_PARALLELISM_DIM_0 = 4,
    parameter WEIGHT_PARALLELISM_DIM_1 = 4,
    parameter WEIGHT_PARALLELISM_DIM_2 = 4,
    parameter WEIGHT_PARALLELISM_DIM_3 = 4,


    parameter BIAS_PRECISION_0       = 8,
    parameter BIAS_PRECISION_1       = 4,
    parameter BIAS_TENSOR_SIZE_DIM_0 = 4,
    parameter BIAS_TENSOR_SIZE_DIM_1 = 4,
    parameter BIAS_PARALLELISM_DIM_0 = 4,
    parameter BIAS_PARALLELISM_DIM_1 = 4,

    parameter PADDING_TENSOR_SIZE_DIM_0_VALUE = 1,
    parameter PADDING_TENSOR_SIZE_DIM_1_VALUE = 1,
    parameter STRIDE_TENSOR_SIZE_DIM_0_VALUE  = 1,
    parameter STRIDE_TENSOR_SIZE_DIM_1_VALUE  = 1,

    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 3,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 3,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_2 = 2,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_3 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 4,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 4,
    parameter DATA_OUT_0_PARALLELISM_DIM_2 = 4,
    parameter DATA_OUT_0_PARALLELISM_DIM_3 = 4,
    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 4,

    // Weight shape (2,1,3,3)
    // 2: number of output channel, i.e. determines number of filters we have
    // E.g. if shape = (2,2,3,3) since the first number is 2, it means we still have 2 numbers
    // 1: number of input channel
    // 1st 3: height
    // 2nd 3: width
    // Then after hardware pass, dim_0 and 1 refers to 3,3; dim_2 refers to 1, 3 refers to 2
    // Number of parameters if weight shape is (2,2,3,3): 2x2x3x3=36
    // i.e. 2 filters (2nd in shape), since 2 input channel (1st in shape), each filter has distinct
    // parameters to each input channel, i.e. each filter would have 2(first in shape)x3x3=18 parameters

    // Input Shape: (batch size, input channel number, height, width)
    // This means IN_C=DATA_IN_0_TENSOR_SIZE_DIM_2 =WEIGHT_TENSOR_SIZE_DIM_2

    // In_X and In_Y the data width and height
    // In_C number of input channels
    // Input CNN shape is [batch_size,channel number,yheight,xwidth]
    parameter IN_X = 4,
    parameter IN_Y = 4,
    parameter IN_C = 1,  // IN_C=DATA_IN_0_TENSOR_SIZE_DIM_2 =WEIGHT_TENSOR_SIZE_DIM_2// They need to match
    parameter UNROLL_IN_C = 1,  //how many input channels are processed in parallel


    parameter KERNEL_X = 3,  //WEIGHT_TENSOR_SIZE_DIM_0
    parameter KERNEL_Y = 3,  //WEIGHT_TENSOR_SIZE_DIM_1
    parameter OUT_C = 1,  // number of filters->WEIGHT_TENSOR_SIZE_DIM_3

    // UNROLL_KERNEL_OUT tells the hardware how many kernel multiplications
    // (from the current dot product for each output channel) are performed in parallel
    // during each “slice” of the convolution arithmetic.
    // E.g. Suppose each window of the input has 3×3 = 9 positions, a
    // and you process 2 input channels in parallel (UNROLL_IN_C = 2).
    // That means ROLL_IN_NUM=3×3×2=18
    // If UNROLL_KERNEL_OUT = 4, then you do 4 multiplications in parallel at each step. To handle all 18 multiplications:
    // Cycle 1: do multiplications #1–#4
    // Cycle 2: do multiplications #5–#8
    // Cycle 3: do multiplications #9–#12
    // Cycle 4: do multiplications #13–#16
    // Cycle 5: do multiplications #17–#18 (only 2 of the 4 “slots” are used here)
    // After cycle 5, you have accumulated the entire dot product for that output channel.

    parameter UNROLL_KERNEL_OUT = 9,
    parameter UNROLL_OUT_C = 1,
    // UNROLL_OUT_CL controls how many output channels are generated (or updated) in parallel
    // each cycle (each channel generates one pixel per location )


    parameter SLIDING_NUM = 8,

    parameter BIAS_SIZE = UNROLL_OUT_C,
    parameter STRIDE    = 1,

    parameter PADDING_Y = 1,
    parameter PADDING_X = 2


) (
    input clk,
    input rst,

    input  [DATA_IN_0_PRECISION_0 - 1:0] data_in_0      [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1*DATA_IN_0_PARALLELISM_DIM_2*DATA_IN_0_PARALLELISM_DIM_3- 1 : 0],  //2
    input data_in_0_valid,
    output data_in_0_ready,

    input  [WEIGHT_PRECISION_0-1:0] weight      [WEIGHT_PARALLELISM_DIM_0*WEIGHT_PARALLELISM_DIM_1*WEIGHT_PARALLELISM_DIM_2*WEIGHT_PARALLELISM_DIM_3 -1:0],
    input weight_valid,
    output weight_ready,

    input  [BIAS_PRECISION_0-1:0] bias      [BIAS_PARALLELISM_DIM_0*BIAS_PARALLELISM_DIM_1-1:0],
    input                         bias_valid,
    output                        bias_ready,

    output [DATA_OUT_0_PRECISION_0 - 1:0] data_out_0      [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1*DATA_OUT_0_PARALLELISM_DIM_2*DATA_OUT_0_PARALLELISM_DIM_3 - 1:0],
    output data_out_0_valid,
    input data_out_0_ready
);
  initial begin
    assert (((UNROLL_IN_C * KERNEL_X * KERNEL_Y) % UNROLL_KERNEL_OUT == 0) & ((UNROLL_IN_C * KERNEL_X * KERNEL_Y) >= UNROLL_KERNEL_OUT) & (UNROLL_IN_C <= IN_C) & (UNROLL_OUT_C <= OUT_C) & (IN_C % UNROLL_IN_C==0)&(OUT_C % UNROLL_OUT_C==0))
    else $fatal("UNROLL parameter not set correctly");
  end



  // Define internal logic for unroll data
  logic [ DATA_IN_0_PRECISION_0 - 1:0] sliding_data_in_0  [                  UNROLL_IN_C - 1 : 0];
  logic [      WEIGHT_PRECISION_0-1:0] internal_weight    [UNROLL_KERNEL_OUT * UNROLL_OUT_C -1:0];
  logic [        BIAS_PRECISION_0-1:0] internal_bias      [                        BIAS_SIZE-1:0];
  logic [DATA_OUT_0_PRECISION_0 - 1:0] internal_data_out_0[                   UNROLL_OUT_C - 1:0];

  localparam UNCAST_OUT_WIDTH = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
      KERNEL_Y * KERNEL_X * IN_C
  ) + 1;
  localparam UNCAST_OUT_FRAC_WIDTH = DATA_IN_0_PRECISION_1 + WEIGHT_PRECISION_1;
  logic [DATA_IN_0_PRECISION_0 * UNROLL_IN_C - 1:0] packed_data_in;
  logic [UNCAST_OUT_WIDTH - 1:0] uncast_data_out[UNROLL_OUT_C - 1:0];

  // Tells how many input data it needs to process for unroll number of channels when it passes into roller
  // e.g. when weight (2,2,3,3), unroll_IN_C=2 (process two input channels in parallel), it means
  // the data it needs is 3x3x2=18
  localparam ROLL_IN_NUM = KERNEL_Y * KERNEL_X * UNROLL_IN_C;


  logic [DATA_IN_0_PRECISION_0 - 1:0] rolled_k[UNROLL_KERNEL_OUT - 1:0];
  logic rolled_k_valid;
  logic rolled_k_ready;
  for (genvar i = 0; i < UNROLL_IN_C; i++)
  for (genvar j = 0; j < DATA_IN_0_PRECISION_0; j++)
    assign packed_data_in[i*DATA_IN_0_PRECISION_0+j] = sliding_data_in_0[i][j];

  logic [DATA_IN_0_PRECISION_0 * UNROLL_IN_C - 1:0] packed_kernel[KERNEL_Y * KERNEL_X - 1:0];
  // in the array packed_kernel whose size is defined by filter size
  // each element in the array packed_kernel[i] stores UNROLL_IN_C channels worth of data:
  // packed_kernel[i] = { channel_data[UNROLL_IN_C-1], ..., channel_data[1], channel_data[0] }
  // the size of each packed_kernel[i] tells the total number of bits for unroll_in_c number of pixels
  logic [DATA_IN_0_PRECISION_0 - 1:0] kernel[KERNEL_Y * KERNEL_X * UNROLL_IN_C - 1:0];
  logic kernel_valid;
  logic kernel_ready;


  data_in_reshaper #(
      .DATA_WIDTH     (DATA_IN_0_PRECISION_0),        // Data width
      .IMG_WIDTH      (DATA_IN_0_TENSOR_SIZE_DIM_0),  // Image width
      .IMG_HEIGHT     (DATA_IN_0_TENSOR_SIZE_DIM_1),  // Image height
      .IN_CHANNELS    (DATA_IN_0_TENSOR_SIZE_DIM_2),  // Input channels
      .BATCH_SIZE     (DATA_IN_0_TENSOR_SIZE_DIM_3),  // Number of batches
      .UNROLL_CHANNELS(UNROLL_IN_C),                  // Number of channels processed in parallel
      .SPATIAL_GROUP_X(DATA_IN_0_PARALLELISM_DIM_0),  // X dimension of spatial groups
      .SPATIAL_GROUP_Y(DATA_IN_0_PARALLELISM_DIM_1)
  ) data_in_reshaper_0 (
      .clk(clk),
      .rst(rst),
      .data_in(data_in_0),
      .data_in_valid(data_in_0_valid),
      .data_in_ready(data_in_0_ready),
      .data_out_ready(reshape_ready),
      .data_out_valid(reshape_valid),
      .data_out(sliding_data_in_0)
  );

  logic reshape_valid;
  logic reshape_ready;


  //   Function of sliding window:
  // You feed it a stream of data, where each clock cycle provides up to UNROLL_IN_C pixels (data_in_0),
  // each pixel is from different channels and being DATA_IN_0_PRECISION_0 bits wide.
  // The sliding window accumulates these pixels in a set of internal buffers
  // Once it collects enough data:
  // The sub-module’s output is an array of size [KERNEL_X * KERNEL_Y].
  // Each element in that array is a wide word of bit-width DATA_IN_0_PRECISION_0 * UNROLL_IN_C.

  logic [DATA_IN_0_PRECISION_0 - 1:0] sliding_data_in_0_single;
  assign sliding_data_in_0_single = sliding_data_in_0[0];
  logic padding_data_out_valid;
  logic padding_data_out_ready;
  logic [DATA_IN_0_PRECISION_0 - 1:0] padding_mase_data_out;
  logic striding_ready;
  assign striding_ready_test = 'b1;
  padding_mase #(
      .UNROLL(UNROLL_IN_C),
      .PADDING(PADDING_TENSOR_SIZE_DIM_0_VALUE),
      .DATA_IN_PRECISION_0(DATA_IN_0_PRECISION_0),
      .DATA_IN_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
      .DATA_IN_0_TENSOR_SIZE_DIM_1(DATA_IN_0_TENSOR_SIZE_DIM_1)
  ) padding_mase_inst1 (
      .clk(clk),
      .rst(rst),
      .data_in(sliding_data_in_0_single),
      .data_in_valid(reshape_valid),
      .data_in_ready(reshape_ready),
      .data_out(padding_mase_data_out),
      .data_out_valid(padding_data_out_valid),
      .data_out_ready(striding_ready)
  );

  logic [DATA_IN_0_PRECISION_0 - 1:0] striding_data_out [WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0];
  logic striding_data_out_valid;
  //   assign conv_arith_ready = 'd1;
  striding #(
      .DATA_WIDTH(DATA_IN_0_PRECISION_0),
      .KERNEL_X(WEIGHT_TENSOR_SIZE_DIM_0),
      .KERNEL_Y(WEIGHT_TENSOR_SIZE_DIM_1),
      .DATA_IN_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
      .DATA_IN_0_TENSOR_SIZE_DIM_1(DATA_IN_0_TENSOR_SIZE_DIM_1),
      .PADDING(PADDING_TENSOR_SIZE_DIM_0_VALUE)
  ) striding_inst1 (
      .clk(clk),
      .rst_n(rst),
      .pixel_in(padding_mase_data_out),
      .pixel_in_valid(padding_data_out_valid),
      .pixel_in_ready(striding_ready),
      .result_out(striding_data_out),
      .sliding_window_valid(striding_data_out_valid),
      .sliding_window_ready(conv_arith_ready)
  );


  // needs to be checked
  logic [WEIGHT_PRECISION_0-1:0] buffer_weight_out [WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0];
  logic buffer_valid;

  weight_buffer #(
      .WEIGHT_TENSOR_SIZE_DIM_0(WEIGHT_TENSOR_SIZE_DIM_0),
      .WEIGHT_TENSOR_SIZE_DIM_1(WEIGHT_TENSOR_SIZE_DIM_1),
      .WEIGHT_PRECISION_0(WEIGHT_PRECISION_0),
      .WEIGHT_PARALLELISM_DIM_0(WEIGHT_PARALLELISM_DIM_0),
      .WEIGHT_PARALLELISM_DIM_1(WEIGHT_PARALLELISM_DIM_1)
  ) weight_buffer_inst (
      .clk(clk),
      .rst(rst),
      .weight_valid(weight_valid),  // top gives to here
      .weight_ready(weight_ready),  // gives to weight_source module to send data
      .weight_data(weight),
      .buffer_valid(buffer_valid),
      .buffer_ready(weight_math_ready),
      .buffer_out(buffer_weight_out)
  );


  logic arith_ready = 'b1;
  localparam ARITH_DATA_OUT_WIDTH = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
      WEIGHT_TENSOR_SIZE_DIM_0 * WEIGHT_TENSOR_SIZE_DIM_1
  );

  logic [ARITH_DATA_OUT_WIDTH-1:0] arith_data_out;
  conv_arith_mase #(
      .DATA_IN_0_PRECISION_0(DATA_IN_0_PRECISION_0),
      .WEIGHT_PRECISION_0(WEIGHT_PRECISION_0),
      .WEIGHT_TENSOR_SIZE_DIM_0(WEIGHT_TENSOR_SIZE_DIM_0),
      .WEIGHT_TENSOR_SIZE_DIM_1(WEIGHT_TENSOR_SIZE_DIM_1)
  ) conv_arith_inst (
      .clk(clk),
      .rst(rst),
      .weight_valid(buffer_valid),  // top gives to here
      .weight_ready(weight_math_ready),  // gives to weight_source module to send data
      .weight_data(buffer_weight_out),
      .data_in_valid(striding_data_out_valid),
      .data_in_ready(conv_arith_ready),
      .data_in(striding_data_out),
      .arith_valid(arith_valid),
      .arith_ready(arith_ready),
      .arith_data_out(arith_data_out)
  );

  fixed_signed_cast #(
      .IN_WIDTH(ROUND_PRECISION_0),
      .IN_FRAC_WIDTH(ROUND_PRECISION_1),
      .OUT_WIDTH(DATA_OUT_0_PRECISION_0),
      .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1),
      .ROUND_FLOOR(1)
  ) fr_inst (
      .in_data (arith_data_out),
      .out_data(fixed_rounding_out)
  );

  localparam ROUND_PRECISION_0 = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
      KERNEL_X * KERNEL_Y * IN_C
  );
  localparam ROUND_PRECISION_1 = DATA_IN_0_PRECISION_1 + WEIGHT_PRECISION_1;
  //   small_convolution_arith #(
  //       // assume output will only unroll_out_channels
  //       .DATA_IN_0_PRECISION_0(DATA_IN_0_PRECISION_0),
  //       .DATA_IN_0_PRECISION_1(DATA_IN_0_PRECISION_1),
  //       .WEIGHT_PRECISION_0(WEIGHT_PRECISION_0),
  //       .WEIGHT_PRECISION_1(WEIGHT_PRECISION_1),
  //       .BIAS_PRECISION_0(BIAS_PRECISION_0),
  //       .BIAS_PRECISION_1(BIAS_PRECISION_1),
  //       .ROLL_IN_NUM(ROLL_IN_NUM),
  //       .ROLL_OUT_NUM(UNROLL_KERNEL_OUT),
  //       .IN_CHANNELS_DEPTH(DATA_IN_0_TENSOR_SIZE_DIM_3),
  //       .OUT_CHANNELS_PARALLELISM(DATA_OUT_0_PARALLELISM_DIM_2),
  //       //   .OUT_CHANNELS_DEPTH(OUT_C / UNROLL_OUT_C),
  //       .HAS_BIAS(HAS_BIAS)
  //   ) simple_convolution_arith_inst1 (
  //       .data_in_0(striding_data_out),
  //       .data_in_0_valid(striding_data_out_valid),
  //       .data_in_0_ready(conv_arith_ready),  //striding gives to here
  //       .data_out_0(round_in),
  //       .weight(buffer_weight_out),
  //       .weight_ready(arith_ready),
  //       .weight_valid(buffer_valid),
  //       .*
  //   );

  //   assign arith_ready = 'b1;
























  // Since I have packed_kernel
  // I unpacked the kernel, ie. for each array of packed_kernel that contains
  // all pixels of all unroll_in_c channels at that x,y position,
  // I dismantle it and puts each pixel for different channel as single element in kernel array
  /* verilator lint_on PINMISSING */
  //   for (genvar i = 0; i < KERNEL_Y * KERNEL_X; i++)
  //   for (genvar j = 0; j < UNROLL_IN_C; j++)
  //   for (genvar k = 0; k < DATA_IN_0_PRECISION_0; k++)
  //     assign kernel[i*UNROLL_IN_C+j][k] = packed_kernel[i][j*DATA_IN_0_PRECISION_0+k];

  //   //  Tells the roller how many of these pixels (and their corresponding weights)
  //   //  to send to the arith each clock. This is why rolled_k is array of [UNROLL_KERNEL_OUT-1:0],
  //   //  as it sends these num of arrays for matrix at one time.
  //   roller #(
  //       .DATA_WIDTH(DATA_IN_0_PRECISION_0),
  //       .NUM(ROLL_IN_NUM),
  //       .ROLL_NUM(UNROLL_KERNEL_OUT)
  //   ) roller_inst (
  //       .data_in(kernel),
  //       .data_in_valid(kernel_valid),
  //       .data_in_ready(kernel_ready),
  //       .data_out(rolled_k),
  //       .data_out_valid(rolled_k_valid),
  //       .data_out_ready(rolled_k_ready),
  //       .*
  //   );

  //   convolution_arith #(
  //       // assume output will only unroll_out_channels
  //       .DATA_IN_0_PRECISION_0(DATA_IN_0_PRECISION_0),
  //       .DATA_IN_0_PRECISION_1(DATA_IN_0_PRECISION_1),
  //       .WEIGHT_PRECISION_0(WEIGHT_PRECISION_0),
  //       .WEIGHT_PRECISION_1(WEIGHT_PRECISION_1),
  //       .BIAS_PRECISION_0(BIAS_PRECISION_0),
  //       .BIAS_PRECISION_1(BIAS_PRECISION_1),
  //       .ROLL_IN_NUM(ROLL_IN_NUM),
  //       .ROLL_OUT_NUM(UNROLL_KERNEL_OUT),
  //       .IN_CHANNELS_DEPTH(IN_C / UNROLL_IN_C),
  //       .OUT_CHANNELS_PARALLELISM(UNROLL_OUT_C),
  //       .OUT_CHANNELS_DEPTH(OUT_C / UNROLL_OUT_C),
  //       .WEIGHT_REPEATS(SLIDING_NUM),
  //       .HAS_BIAS(HAS_BIAS)
  //   ) convolution_arith_inst (
  //       .data_in_0(rolled_k),
  //       .data_in_0_valid(rolled_k_valid),
  //       .data_in_0_ready(rolled_k_ready),
  //       .data_out_0(round_in),
  //       .weight(internal_weight)
  //   );

  //   fixed_rounding #(
  //       .IN_SIZE(UNROLL_OUT_C),
  //       .IN_WIDTH(ROUND_PRECISION_0),
  //       .IN_FRAC_WIDTH(ROUND_PRECISION_1),
  //       .OUT_WIDTH(DATA_OUT_0_PRECISION_0),
  //       .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1)
  //   ) round_inst (
  //       .data_in (round_in),
  //       .data_out(internal_data_out_0)
  //   );

endmodule
