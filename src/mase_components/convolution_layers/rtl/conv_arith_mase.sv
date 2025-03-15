module conv_arith_mase #(
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter WEIGHT_PRECISION_0      = 8,
    parameter WEIGHT_TENSOR_SIZE_DIM_0 = 3,
    parameter WEIGHT_TENSOR_SIZE_DIM_1 = 3,
    // ARITH_DATA_OUT_WIDTH = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(3*3)
    parameter ARITH_DATA_OUT_WIDTH    = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
        WEIGHT_TENSOR_SIZE_DIM_0 * WEIGHT_TENSOR_SIZE_DIM_1
    )
) (
    input  logic clk,
    input  logic rst,

    // Weight interface
    input  logic weight_valid,
    output logic weight_ready,
    input  logic [WEIGHT_PRECISION_0-1:0] weight_data[WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0],

    // Data in interface
    input  logic data_in_valid,
    output logic data_in_ready,
    input  logic [DATA_IN_0_PRECISION_0-1:0] data_in[WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0],

    // Arithmetic result interface
    output logic arith_valid,
    input  logic arith_ready,
    output logic [ARITH_DATA_OUT_WIDTH-1:0] arith_data_out
);

  // Local parameter for total elements (e.g., 3x3 = 9)
  localparam int TOTAL_ELEMENTS = WEIGHT_TENSOR_SIZE_DIM_0 * WEIGHT_TENSOR_SIZE_DIM_1;

  // State encoding for a simple FSM
  typedef enum logic [1:0] {
    IDLE,
    ACCUM,
    DONE
  } state_t;
  state_t state;

  // Internal registers to hold the input arrays
  logic signed [WEIGHT_PRECISION_0-1:0] weight_reg [0:TOTAL_ELEMENTS-1];
  logic signed [DATA_IN_0_PRECISION_0-1:0] data_reg   [0:TOTAL_ELEMENTS-1];
  logic signed [ARITH_DATA_OUT_WIDTH-1:0]   accumulator;

  // Counter for processing elements; width is enough to count up to TOTAL_ELEMENTS (9)
  logic [$clog2(TOTAL_ELEMENTS+1)-1:0] counter;

  // Output assignment: arithmetic output reflects the accumulator value.
  assign arith_data_out = accumulator;

  // Combinational handshake for weight and data inputs:
  // Only when in IDLE and the complementary valid is high.
  assign weight_ready = (state == IDLE) && data_in_valid;
  assign data_in_ready = (state == IDLE) && weight_valid;
  
  // Combinational handshake for arithmetic output:
  // arith_valid is asserted when the state machine is in DONE.
  assign arith_valid = (state == DONE);

  // FSM and sequential multiply-accumulate process.
  always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
      state       <= IDLE;
      counter     <= 0;
      accumulator <= 0;
    end else begin
      case (state)
        IDLE: begin
          // Wait until both inputs are valid
          if (weight_valid && data_in_valid) begin
            // Capture the entire weight and data arrays.
            for (int i = 0; i < TOTAL_ELEMENTS; i++) begin
              weight_reg[i] <= weight_data[i];
              data_reg[i]   <= data_in[i];
            end
            // Initialize accumulator with the first multiplication result.
            accumulator <= $signed(weight_data[0]) * $signed(data_in[0]);
            counter     <= 1;
            state       <= ACCUM;
          end
        end

        ACCUM: begin
          // Process remaining elements one per clock cycle.
          if (counter < TOTAL_ELEMENTS) begin
            accumulator <= accumulator +
              ($signed(weight_reg[counter]) * $signed(data_reg[counter]));
            counter <= counter + 1;
          end else begin
            state <= DONE;
          end
        end

        DONE: begin
          // When arith_ready is asserted, transition back to IDLE.
          if (arith_ready) begin
            state <= IDLE;
          end
        end

        default: state <= IDLE;
      endcase
    end
  end

endmodule
