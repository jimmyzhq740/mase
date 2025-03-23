
module conv_arith_mase_array #(
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_PRECISION_1 = 5,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 4,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 4,
    parameter WEIGHT_TENSOR_SIZE_DIM_0 = 3,
    parameter WEIGHT_TENSOR_SIZE_DIM_1 = 3,
    parameter WEIGHT_TENSOR_SIZE_DIM_3 = 4,  // how many output channel
    parameter WEIGHT_PRECISION_0 = 8,
    parameter WEIGHT_PRECISION_1 = 5,
    parameter WEIGHT_PARALLELISM_DIM_0 = 2,
    parameter WEIGHT_PARALLELISM_DIM_1 = 2,
    parameter WEIGHT_PARALLELISM_DIM_3 = 2,  // total array size
    // Data_out_parallelism_2 used to determine how many weights array is used to calculate with the inputs
    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 5,
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 2,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 2,
    parameter DATA_OUT_0_PARALLELISM_DIM_2 = 2,
    parameter ARITH_DATA_OUT_WIDTH = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
        WEIGHT_TENSOR_SIZE_DIM_0 * WEIGHT_TENSOR_SIZE_DIM_1
    ),
    parameter OUTPUT_SIZE = DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1*DATA_OUT_0_PARALLELISM_DIM_2

) (
    input logic clk,
    input logic rst,
    input logic [WEIGHT_PRECISION_0*WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0] weight_data_mul[WEIGHT_TENSOR_SIZE_DIM_3-1:0],


    // Weight interface
    input  logic weight_valid,
    output logic weight_ready,
    //input  logic [WEIGHT_PRECISION_0-1:0] weight_data[WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0],
    // single element as I have another layer to build multiple of this block
    //input logic [WEIGHT_PRECISION_0*WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0] weight_data_mul,

    // Data in interface
    input logic data_in_valid,
    output logic data_in_ready,
    input  logic [DATA_IN_0_PRECISION_0-1:0] data_in[WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0],

    // Arithmetic result interface
    output logic arith_parallel_valid,
    input logic arith_ready,
    output logic [DATA_OUT_0_PRECISION_0-1:0] arith_parallel_data_out[OUTPUT_SIZE-1:0],
    output logic [ARITH_DATA_OUT_WIDTH-1:0] arith_data_out

);

  logic [DATA_IN_0_PRECISION_0*WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0] buffer_array_mul [DATA_IN_0_TENSOR_SIZE_DIM_0*DATA_IN_0_TENSOR_SIZE_DIM_1-1:0];
  logic striding_input_buffer_out_valid;
  // store all striding inputs for not first channel calculation
  striding_input_buffer #(
      .DATA_IN_0_TENSOR_SIZE_DIM_0(DATA_IN_0_TENSOR_SIZE_DIM_0),
      .DATA_IN_0_TENSOR_SIZE_DIM_1(DATA_IN_0_TENSOR_SIZE_DIM_1),
      .WEIGHT_TENSOR_SIZE_DIM_0(WEIGHT_TENSOR_SIZE_DIM_0),
      .WEIGHT_TENSOR_SIZE_DIM_1(WEIGHT_TENSOR_SIZE_DIM_1), // striding input tensor depends on weight tensor dim size
      .DATA_IN_0_PRECISION_0(DATA_IN_0_PRECISION_0)
  ) striding_input_buffer_inst (
      .clk(clk),
      .rst(rst),
      .data_in_valid(selected_data_in_valid),
      .buffer_ready(data_in_ready),  // from conv_arith_mase to tell when the next comes in
      // .data_in_ready(data_in_ready), // dont need to tell outside world it's ready
      .striding_data_in(data_in),
      .buffer_valid(striding_input_buffer_out_valid),
      .buffer_out_mul(buffer_array_mul)
  );


  // MUX signal: select between live data_in and buffered data.
  // When in the S_COMPUTE_DIRECT state, use the live data_in; otherwise, use the buffered data.
  logic [DATA_IN_0_PRECISION_0-1:0] selected_data_in [WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0];
  logic [DATA_IN_0_PRECISION_0-1:0] data_from_striding_buffer [WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0];

  // when the striding data_in is valid, we use weights array to calculate with data_in->out_parallelism defines
  // how many arrays used. Conv_arith_mase is the arith block, so out_parallelism define
  // how many this block is used in parallel at one calculation period

  // Instantiate N parallel conv_arith_mase blocks
  // input tensor_0 and 1 tells us how many striding output is (assume padding is 1), which tells how many computation needed for 1 output channel

  // Have to change this since every clk it updates, I need to think a way when the first two parallel_dim2 finishes
  // Then calculate next two
  //channel_counter-> counts which channel to be calculated
  // Counter to keep track of your current position in weight_array_in
  // Introudce a selected_data_in_valid to give to arith to let it know the datain is valid to calculate
  logic selected_data_in_valid;
  logic [$clog2(WEIGHT_TENSOR_SIZE_DIM_3)-1:0] channel_counter;

  // TOTAL striding size each time:
  localparam TOTAL_STRIDING = WEIGHT_TENSOR_SIZE_DIM_0 * WEIGHT_TENSOR_SIZE_DIM_1;
  // How many striding inputs
  localparam TOTAL_STRIDING_INPUT = DATA_IN_0_TENSOR_SIZE_DIM_0 * DATA_IN_0_TENSOR_SIZE_DIM_1;
  // used to track which striding input from the buffer
  logic [$clog2(TOTAL_STRIDING_INPUT)-1:0] striding_data_counter = TOTAL_STRIDING_INPUT - 1;

  typedef enum logic [1:0] {
    IDLE,
    DATA_IN_LIVE,
    DATA_IN_BUFFER
  } state_t;
  state_t state;

  logic   idle;
  logic   data_in_live;
  logic   data_in_buffer;

  always_ff @(posedge clk) begin
    if (rst) begin
      state <= IDLE;
      channel_counter <= 0;
      assign selected_data_in_valid = 'b0;
    end
    case (state)
      IDLE: begin
        assign selected_data_in_valid = 'b0;  // IDLE state: dont let arith to start
        striding_data_counter <= TOTAL_STRIDING_INPUT - 1;
        if (data_in_valid && channel_counter == 0) begin
          assign selected_data_in = data_in;
          assign selected_data_in_valid = 'b1;
          state <= DATA_IN_LIVE;
          data_in_live <= 1;
          data_in_buffer <= 0;
        end
        if (channel_counter>= DATA_OUT_0_PARALLELISM_DIM_2&&striding_input_buffer_out_valid) begin
          state <= DATA_IN_BUFFER;
          data_in_buffer <= 1;
          data_in_live <= 0;
          idle <= 0;

        end
      end
      DATA_IN_LIVE: begin
        state <= DATA_IN_LIVE;
        data_in_live <= 1;

        if (data_in_valid) begin
          assign selected_data_in = data_in;
          assign selected_data_in_valid = 'b1;
        end else begin
          assign selected_data_in_valid = 'b0;
        end
        if (image_done) begin
          channel_counter <= channel_counter + DATA_OUT_0_PARALLELISM_DIM_2;
          state <= IDLE;
          idle <= 1;
        end
      end
      DATA_IN_BUFFER: begin
        state <= DATA_IN_BUFFER;
        data_in_buffer <= 1;

        if (channel_counter>= DATA_OUT_0_PARALLELISM_DIM_2&&striding_input_buffer_out_valid) begin

          if (data_in_ready) begin
            for (int i = 0; i < TOTAL_STRIDING; i++) begin
              data_from_striding_buffer[TOTAL_STRIDING-i-1] = buffer_array_mul[striding_data_counter][(TOTAL_STRIDING-i)*DATA_IN_0_PRECISION_0-1 -: DATA_IN_0_PRECISION_0];
            end

            assign selected_data_in_valid = 'b1;
            assign selected_data_in = data_from_striding_buffer;

            striding_data_counter <= striding_data_counter - 1;
          end else begin
            assign selected_data_in_valid = 'b0;
          end

        end
        if (image_done) begin
          channel_counter <= channel_counter + DATA_OUT_0_PARALLELISM_DIM_2;
          assign selected_data_in_valid = 'b0;
          state <= IDLE;
          data_in_buffer <= 0;
          idle <= 1;
        end
        if (channel_counter == WEIGHT_TENSOR_SIZE_DIM_3) begin
          assign selected_data_in_valid = 'b0;
          state <= IDLE;
          idle  <= 1;
        end
      end
    endcase
  end

  localparam IN_C = 'b1;
  logic [ ARITH_DATA_OUT_WIDTH-1:0] debug_arith_data_out[DATA_OUT_0_PARALLELISM_DIM_2-1:0];
  logic [DATA_IN_0_PRECISION_0-1:0] fixed_rounding_out  [DATA_OUT_0_PARALLELISM_DIM_2-1:0];
  localparam ROUND_PRECISION_0 = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
      WEIGHT_TENSOR_SIZE_DIM_0 * WEIGHT_TENSOR_SIZE_DIM_1 * IN_C
  );
  localparam ROUND_PRECISION_1 = DATA_IN_0_PRECISION_1 + WEIGHT_PRECISION_1;
  logic image_done;
  logic arith_valid;
  genvar i;
  generate
    for (i = 0; i < DATA_OUT_0_PARALLELISM_DIM_2; i++) begin : parallel_blocks
      conv_arith_mase #(
          .DATA_IN_0_PRECISION_0(DATA_IN_0_PRECISION_0),
          .WEIGHT_PRECISION_0(WEIGHT_PRECISION_0),
          .WEIGHT_TENSOR_SIZE_DIM_0(WEIGHT_TENSOR_SIZE_DIM_0),
          .WEIGHT_TENSOR_SIZE_DIM_1(WEIGHT_TENSOR_SIZE_DIM_1),
          .WEIGHT_TENSOR_SIZE_DIM_3(WEIGHT_TENSOR_SIZE_DIM_3)
      ) u_conv (
          .clk(clk),
          .rst(rst),
          .weight_valid(weight_valid),  // top gives to here
          .weight_ready(weight_ready),
          .weight_data_mul(weight_data_mul[WEIGHT_TENSOR_SIZE_DIM_3-1-(channel_counter+i)]),
          .data_in_valid(selected_data_in_valid),
          .data_in_ready(data_in_ready),
          .data_in(selected_data_in),
          .arith_valid(arith_valid),
          .arith_ready(arith_ready),
          .arith_data_out(debug_arith_data_out[DATA_OUT_0_PARALLELISM_DIM_2-i-1]),
          .image_done(image_done)
      );
      fixed_signed_cast #(
          .IN_WIDTH(ROUND_PRECISION_0),
          .IN_FRAC_WIDTH(ROUND_PRECISION_1),
          .OUT_WIDTH(DATA_OUT_0_PRECISION_0),
          .OUT_FRAC_WIDTH(DATA_OUT_0_PRECISION_1),
          .ROUND_FLOOR(1)
      ) fr_inst (
          .in_data (debug_arith_data_out[DATA_OUT_0_PARALLELISM_DIM_2-i-1]),
          .out_data(fixed_rounding_out[DATA_OUT_0_PARALLELISM_DIM_2-i-1])
      );


    end
  endgenerate

  // localparam OUTPUT_SIZE = DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1*DATA_OUT_0_PARALLELISM_DIM_2;
  localparam OUT_SIZE_EACH_CHANNEL = DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1;
  logic [DATA_OUT_0_PRECISION_0-1:0] out_buffer[OUTPUT_SIZE-1:0];
  logic [$clog2(OUTPUT_SIZE)-1:0] out_counter;
  logic out_buffer_valid;

  always_ff @(posedge clk) begin
    // when the calculation for dim_2 number of channel pixels finish,
    // i.e. arith_valid ==1, then store dim0*dim1*dim2 number to sent out
    assign out_buffer_valid = 'b0;
    if (arith_valid) begin
      out_counter <= out_counter + 1;
      for (int i = 0; i < DATA_OUT_0_PARALLELISM_DIM_2; i++) begin
        out_buffer[OUTPUT_SIZE-1-(OUT_SIZE_EACH_CHANNEL*i)-out_counter]=fixed_rounding_out[DATA_OUT_0_PARALLELISM_DIM_2-1-i];
      end
    end
    if (out_counter == (OUT_SIZE_EACH_CHANNEL)) begin
      out_counter <= 0;
      assign out_buffer_valid = 'b1;
    end
  end

  assign arith_parallel_valid = out_buffer_valid;
  assign arith_parallel_data_out = out_buffer;

endmodule

// # 64, -65, -96, 192,
// # -10, 168, -84, -112,
// # -127, -143, -274, 160
// # -81, 28, -70, -64
