module weight_buffer #(
    parameter WEIGHT_TENSOR_SIZE_DIM_0 = 3,
    parameter WEIGHT_TENSOR_SIZE_DIM_1 = 3,
    parameter WEIGHT_PRECISION_0       = 8,
    parameter WEIGHT_PARALLELISM_DIM_0 = 2,
    parameter WEIGHT_PARALLELISM_DIM_1 = 2
) (
    input logic clk,
    input logic rst,  // active-high reset

    // Input handshake interface (receiving from previous module)
    input logic weight_valid,  // source asserts when data is valid
    output logic weight_ready,  // our module asserts when ready to accept data
    input  logic [WEIGHT_PRECISION_0-1:0] weight_data[WEIGHT_PARALLELISM_DIM_0*WEIGHT_PARALLELISM_DIM_1-1:0],

    // Output handshake interface (sending to next module)
    output logic buffer_valid,  // our module asserts when the package is ready
    input logic buffer_ready,  // next module asserts when it is ready to accept the package
    output logic [WEIGHT_PRECISION_0-1:0] buffer_out[WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0]
);


  // Derived parameters
  localparam TOTAL_WEIGHTS = WEIGHT_TENSOR_SIZE_DIM_0 * WEIGHT_TENSOR_SIZE_DIM_1;
  localparam PARALLEL = WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1;
  // Number of transactions needed (ceiling division)
  localparam TRANSACTION_COUNT = (TOTAL_WEIGHTS + PARALLEL - 1) / PARALLEL;

  // Internal registers
  // trans_count counts how many weight transactions have been received.
  logic [$clog2(TRANSACTION_COUNT):0] trans_count;
  // Internal buffer array to store all weights.
  logic [WEIGHT_PRECISION_0-1:0] buffer_array[0:TOTAL_WEIGHTS-1];

  // Drive output from internal buffer
  assign buffer_out   = buffer_array;

  // The module is ready to accept new weights as long as we haven't assembled a complete package.
  assign weight_ready = ~buffer_valid;

  // Sequential logic to receive weight transactions and assemble the complete buffer.
  always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
      trans_count  <= 0;
      buffer_valid <= 0;
      // Optionally: Initialize buffer_array to zeros.
      // for (int j = 0; j < TOTAL_WEIGHTS; j++) begin
      //   buffer_array[j] <= '0;
      // end
    end else begin
      // When the downstream module accepts the complete package,
      // reset our counters and valid flag.
      if (buffer_valid && buffer_ready) begin
        trans_count  <= 0;
        buffer_valid <= 0;
      end  // Otherwise, if new weight data is available, capture it.
      else if (weight_valid && weight_ready) begin
        integer i;
        integer start_idx;
        // Compute the starting index for this transaction
        start_idx = trans_count * PARALLEL;

        // Save each element of the current transaction into the proper location.
        for (i = 0; i < PARALLEL; i = i + 1) begin
          if ((start_idx + i) < TOTAL_WEIGHTS) begin
            buffer_array[start_idx+i] <= weight_data[PARALLEL-1-i];
          end
        end

        // Increment the transaction counter.
        trans_count <= trans_count + 1;

        // Once we've received the last transaction, assert the output handshake.
        if (trans_count == TRANSACTION_COUNT - 1) begin
          buffer_valid <= 1;
        end
      end
    end
  end

endmodule
