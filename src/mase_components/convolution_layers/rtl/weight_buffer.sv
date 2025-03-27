module weight_buffer #(
    parameter WEIGHT_TENSOR_SIZE_DIM_0 = 3,
    parameter WEIGHT_TENSOR_SIZE_DIM_1 = 3,
    parameter WEIGHT_TENSOR_SIZE_DIM_3 = 4,
    parameter WEIGHT_PRECISION_0       = 8,
    parameter WEIGHT_PARALLELISM_DIM_0 = 2,
    parameter WEIGHT_PARALLELISM_DIM_1 = 2,
    parameter WEIGHT_PARALLELISM_DIM_3 = 4
    // dim3 is the output channel size
) (
    input logic clk,
    input logic rst,  // active-high reset

    // Input handshake interface (receiving from previous module)
    input logic weight_valid,  // source asserts when data is valid
    output logic weight_ready,  // our module asserts when ready to accept data
    // How many data in one row in rom
    input  logic [WEIGHT_PRECISION_0-1:0] weight_data[WEIGHT_PARALLELISM_DIM_0*WEIGHT_PARALLELISM_DIM_1*WEIGHT_PARALLELISM_DIM_3-1:0],

    // Output handshake interface (sending to next module)
    output logic buffer_valid,  // our module asserts when the package is ready
    input logic buffer_ready,  // next module asserts when it is ready to accept the package
    output logic [WEIGHT_PRECISION_0-1:0] buffer_out[WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0],
    output logic [WEIGHT_PRECISION_0*WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0] buffer_out_mul [WEIGHT_TENSOR_SIZE_DIM_3-1:0]
);

  // Derived parameters
  // TOTAL weight for each kernel:
  localparam TOTAL_WEIGHTS = WEIGHT_TENSOR_SIZE_DIM_0 * WEIGHT_TENSOR_SIZE_DIM_1;
  //TOTAL weights across whole channel
  localparam TOTAL_WEIGHTS_ARRAY =  WEIGHT_TENSOR_SIZE_DIM_0 * WEIGHT_TENSOR_SIZE_DIM_1*WEIGHT_TENSOR_SIZE_DIM_3;

  // Parallel means how many weights in a line
  localparam PARALLEL = WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1*WEIGHT_PARALLELISM_DIM_3;
  // Number of transactions needed (ceiling division)
  localparam TRANSACTION_COUNT = (TOTAL_WEIGHTS_ARRAY + PARALLEL - 1) / PARALLEL;

  // Internal registers
  // trans_count counts how many weight transactions have been received.
  logic [$clog2(TRANSACTION_COUNT):0] trans_count;
  // Internal buffer array to store all weights.
  logic [WEIGHT_PRECISION_0-1:0] buffer_array[0:TOTAL_WEIGHTS-1];
  // store weights for one kernel in one element (a lot of bits for one element)
  //For weights that are not for the same output channel, store in different element -> hence we store tensor_dim2 arrays
  logic [WEIGHT_PRECISION_0*WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0] buffer_array_mul [WEIGHT_TENSOR_SIZE_DIM_3-1:0];


  // Drive output from internal buffer
  assign buffer_out = buffer_array;
  assign buffer_out_mul = buffer_array_mul;

  // The module is ready to accept new weights only if we haven't assembled a complete package.
  // When buffer_valid is high, weight_ready is 0.
  assign weight_ready = ~buffer_valid;

  // Sequential logic to receive weight transactions and assemble the complete buffer.
  always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
      trans_count  <= 0;
      buffer_valid <= 0;
    end else begin
      // If the package is not yet complete, capture new weight transactions.
      if (!buffer_valid) begin
        if (weight_valid && weight_ready) begin
          integer i;
          integer start_idx;
          integer channel_idx;  // used to know the index number inside the weight_array
          // Compute the starting index for this transaction.

          start_idx = trans_count * PARALLEL;


          // Total_WEIGHTS: How many across weights for each channel
          // Total_WEIGHTS_ARRAY: TOTAL weights across whole channel
          // PARALLEL: how many weights arrive each cycle (e.g., 4).

          // Save each element of the current transaction into the proper location.
          // WEIGHT_TENSOR_SIZE_DIM_3: number of output channels
          // Since every clock cycle, the weights in that line may or may not occupy whole channel
          // start_idx: which line of the code we have
          for (int i = 0; i < PARALLEL; i++) begin
            int global_idx = start_idx + i;
            // if ((start_idx + i) < TOTAL_WEIGHTS) begin
            if (global_idx < TOTAL_WEIGHTS_ARRAY) begin
              // Identify which channel and offset
              // Channel tells me which channel of the weight it currently is
              int channel = global_idx / TOTAL_WEIGHTS;
              // offset_in_channel used to do arithmetic logic for shifting weights in proper place
              int offset_in_channel = global_idx % TOTAL_WEIGHTS;  // remainder e.g. 6%4=2
              int MSB = (TOTAL_WEIGHTS - 1 - offset_in_channel) * WEIGHT_PRECISION_0 + (WEIGHT_PRECISION_0 - 1);
              // each element stores all weights of kernel in that array
              // store all weights of one kernel in one big element
              // start_idx used to indicate when weight order in current channel
              // Introduce a global_idx to indicate track the weight in weight_data
              // e.g. [71 -: 8] means width is 8
              buffer_array[start_idx+i] <= weight_data[PARALLEL-1-i];
              //buffer_array_mul[channel][(TOTAL_WEIGHTS-i-start_idx)*WEIGHT_PRECISION_0-1-:WEIGHT_PRECISION_0] <= weight_data[PARALLEL-1-i];
              // mul[channel]: starts with the rightest, I wanna starts with leftest, i.e. largest: tensor_dim_2-1
              buffer_array_mul[WEIGHT_TENSOR_SIZE_DIM_3-1-channel][MSB-:WEIGHT_PRECISION_0] <= weight_data[PARALLEL-1-i];
              // Debug prints:
              $display("time=%0t | global_idx=%0d | channel=%0d | offset_in_channel=%0d | MSB=%0d",
                       $time, global_idx, channel, offset_in_channel, MSB);
            end
          end

          // Increment the transaction counter.
          trans_count <= trans_count + 1;
          // Once we've received the last transaction, latch the package by asserting buffer_valid.
          if (trans_count == TRANSACTION_COUNT - 1) begin
            // when new channel weights, trans_count becomes 0 for counting that set of trans_count;
            buffer_valid <= 1;
          end
        end
        // Once buffer_valid is high, the module holds the complete package and does not accept new weights.
      end
    end
  end
endmodule


