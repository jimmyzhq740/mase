module striding_input_buffer #(
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 4,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 4,
    parameter WEIGHT_TENSOR_SIZE_DIM_0 = 3,
    parameter WEIGHT_TENSOR_SIZE_DIM_1 = 3, // striding input tensor depends on weight tensor dim size
    parameter DATA_IN_0_PRECISION_0 = 8
    // dim3 is the output channel size
) (
    input logic clk,
    input logic rst,  // active-high reset

    input logic data_in_valid,
    output logic data_in_ready,
    input  logic [DATA_IN_0_PRECISION_0-1:0] striding_data_in[WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0],

    output logic buffer_valid,  // our module asserts when the package is ready
    input logic buffer_ready,  // next module asserts when it is ready to accept the package
    output logic [DATA_IN_0_PRECISION_0-1:0] buffer_out[WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0],
    output logic [DATA_IN_0_PRECISION_0*WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0] buffer_out_mul [DATA_IN_0_TENSOR_SIZE_DIM_0*DATA_IN_0_TENSOR_SIZE_DIM_1-1:0]
);

  // Derived parameters
  // TOTAL striding size each time:
  localparam TOTAL_STRIDING = WEIGHT_TENSOR_SIZE_DIM_0 * WEIGHT_TENSOR_SIZE_DIM_1;
  // How many striding inputs
  localparam TOTAL_STRIDING_INPUT = DATA_IN_0_TENSOR_SIZE_DIM_0 * DATA_IN_0_TENSOR_SIZE_DIM_1;



  logic [$clog2(TOTAL_STRIDING):0] striding_in_count;
  // Internal buffer array to store all weights.
  // store weights for one kernel in one element (a lot of bits for one element)
  //For weights that are not for the same output channel, store in different element -> hence we store tensor_dim2 arrays
  logic [DATA_IN_0_PRECISION_0*WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0] buffer_array_mul [DATA_IN_0_TENSOR_SIZE_DIM_0*DATA_IN_0_TENSOR_SIZE_DIM_1-1:0];


  // Drive output from internal buffer

  assign buffer_out_mul = buffer_array_mul;

  // The module is ready to accept new weights only if we haven't assembled a complete package.
  // When buffer_valid is high, weight_ready is 0.
  assign data_in_ready  = ~buffer_valid;

  // Sequential logic to receive weight transactions and assemble the complete buffer.
  always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
      buffer_valid <= 0;
      striding_in_count <= 0;
    end else begin
      // If the package is not yet complete, capture new weight transactions.
      if (!buffer_valid) begin
        if (data_in_valid && buffer_ready) begin
          integer i;
          // Since striding data_in comes when data_in_valid, so we store this array of striding data_in as single element of an array
          for (int i = 0; i < TOTAL_STRIDING; i++) begin
            // if ((start_idx + i) < TOTAL_WEIGHTS) begin
            //TOTAL_STRIDING: TOTAL striding size each time:
            //TOTAL_STRIDING_INPUT: How many striding inputs
            buffer_array_mul[TOTAL_STRIDING_INPUT-striding_in_count-1][(TOTAL_STRIDING-i)*DATA_IN_0_PRECISION_0-1 -: DATA_IN_0_PRECISION_0] <= striding_data_in[TOTAL_STRIDING-1-i];
            // striding_counter increases by 1 when data_in_valid is 1, i.e. a new striding data array comes in
          end
          striding_in_count <= striding_in_count + 1;


          // Once we've received the last transaction, latch the package by asserting buffer_valid.
          if (striding_in_count == TOTAL_STRIDING_INPUT - 1) begin
            // when new channel weights, trans_count becomes 0 for counting that set of trans_count;
            buffer_valid <= 1;
          end
        end
      end
    end
  end



endmodule

