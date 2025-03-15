module weight_buffer #(
    parameter WEIGHT_TENSOR_SIZE_DIM_0 = 3,
    parameter WEIGHT_TENSOR_SIZE_DIM_1 = 3,
    parameter WEIGHT_PRECISION_0       = 8,
    parameter WEIGHT_PARALLELISM_DIM_0 = 2,
    parameter WEIGHT_PARALLELISM_DIM_1 = 2
) (
    input  logic clk,
    input  logic rst,  // active-high reset

    // Input handshake interface (receiving from previous module)
    input  logic weight_valid,  // source asserts when data is valid
    output logic weight_ready,  // our module asserts when ready to accept data
    input  logic [WEIGHT_PRECISION_0-1:0] weight_data[WEIGHT_PARALLELISM_DIM_0*WEIGHT_PARALLELISM_DIM_1-1:0],

    // Output handshake interface (sending to next module)
    output logic buffer_valid,  // our module asserts when the package is ready
    input  logic buffer_ready,  // next module asserts when it is ready to accept the package
    output logic [WEIGHT_PRECISION_0-1:0] buffer_out[WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0]
);

  // Derived parameters
  localparam TOTAL_WEIGHTS = WEIGHT_TENSOR_SIZE_DIM_0 * WEIGHT_TENSOR_SIZE_DIM_1;
  localparam PARALLEL    = WEIGHT_PARALLELISM_DIM_0 * WEIGHT_PARALLELISM_DIM_1;
  // Number of transactions needed (ceiling division)
  localparam TRANSACTION_COUNT = (TOTAL_WEIGHTS + PARALLEL - 1) / PARALLEL;

  // Internal registers
  // trans_count counts how many weight transactions have been received.
  logic [$clog2(TRANSACTION_COUNT):0] trans_count;
  // Internal buffer array to store all weights.
  logic [WEIGHT_PRECISION_0-1:0] buffer_array[0:TOTAL_WEIGHTS-1];

  // Drive output from internal buffer
  assign buffer_out = buffer_array;

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
          // Compute the starting index for this transaction.
          start_idx = trans_count * PARALLEL;
          // Save each element of the current transaction into the proper location.
          for (i = 0; i < PARALLEL; i = i + 1) begin
            if ((start_idx + i) < TOTAL_WEIGHTS) begin
              buffer_array[start_idx+i] <= weight_data[PARALLEL-1-i];
            end
          end
          // Increment the transaction counter.
          trans_count <= trans_count + 1;
          // Once we've received the last transaction, latch the package by asserting buffer_valid.
          if (trans_count == TRANSACTION_COUNT - 1) begin
            buffer_valid <= 1;
          end
        end
      end
      // Once buffer_valid is high, the module holds the complete package and does not accept new weights.
    end
  end

endmodule

