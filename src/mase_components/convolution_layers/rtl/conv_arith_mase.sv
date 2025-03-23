module conv_arith_mase #(
    parameter DATA_IN_0_PRECISION_0 = 8,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 4,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 4,
    parameter WEIGHT_PRECISION_0 = 8,
    parameter WEIGHT_TENSOR_SIZE_DIM_0 = 3,
    parameter WEIGHT_TENSOR_SIZE_DIM_1 = 3,
    parameter WEIGHT_TENSOR_SIZE_DIM_3 = 1,
    // ARITH_DATA_OUT_WIDTH = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(3*3)
    parameter ARITH_DATA_OUT_WIDTH = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(
        WEIGHT_TENSOR_SIZE_DIM_0 * WEIGHT_TENSOR_SIZE_DIM_1
    )
) (
    input logic clk,
    input logic rst,

    // Weight interface
    input logic weight_valid,
    output logic weight_ready,
    input  logic [WEIGHT_PRECISION_0-1:0] weight_data[WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0],
    // single element as I have another layer to build multiple of this block
    input logic [WEIGHT_PRECISION_0*WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0] weight_data_mul,

    // Data in interface
    input logic data_in_valid,
    output logic data_in_ready,
    input  logic [DATA_IN_0_PRECISION_0-1:0] data_in[WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1-1:0],

    // Arithmetic result interface
    output logic arith_valid,
    input logic arith_ready,
    output logic [ARITH_DATA_OUT_WIDTH-1:0] arith_data_out,
    // outputs when a image with all pixels finishes calculate
    output logic image_done
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
  logic signed [WEIGHT_PRECISION_0-1:0] weight_reg[0:TOTAL_ELEMENTS-1];
  logic signed [WEIGHT_PRECISION_0-1:0] weight_reg_mul[0:TOTAL_ELEMENTS-1];
  logic signed [DATA_IN_0_PRECISION_0-1:0] data_reg[0:TOTAL_ELEMENTS-1];
  logic signed [ARITH_DATA_OUT_WIDTH-1:0] accumulator;
  logic signed [ARITH_DATA_OUT_WIDTH-1:0] accumulator_mul;

  // Counter for processing elements; width is enough to count up to TOTAL_ELEMENTS (9)
  logic [$clog2(TOTAL_ELEMENTS+1)-1:0] counter;
  // pixel counter counts how many pixels needed for one channel
  logic [$clog2(DATA_IN_0_TENSOR_SIZE_DIM_0*DATA_IN_0_TENSOR_SIZE_DIM_0):0] pixel_counter;

  // Output assignment: arithmetic output reflects the accumulator value.
  // I extra added here
  assign arith_data_out = accumulator_mul;

  // Combinational handshake for weight and data inputs:
  // Only when in IDLE and the complementary valid is high.
  assign weight_ready = (state == IDLE) && data_in_valid;
  assign data_in_ready = (state == IDLE) && weight_valid;

  // Combinational handshake for arithmetic output:
  // arith_valid is asserted when the state machine is in DONE.
  assign arith_valid = (state == DONE);

  localparam TOTAL_BITS = WEIGHT_PRECISION_0 * WEIGHT_TENSOR_SIZE_DIM_0 * WEIGHT_TENSOR_SIZE_DIM_1;
  //localparam integer NUM_SLICES = TOTAL_BITS / DATA_IN_0_PRECISION_0; // how many to sweep in the long element of the weight_data

  // FSM and sequential multiply-accumulate process.
  always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
      state           <= IDLE;
      counter         <= 0;
      pixel_counter   <= 0;
      accumulator     <= 0;
      accumulator_mul <= 0;
    end else begin
      if (pixel_counter == DATA_IN_0_TENSOR_SIZE_DIM_0 * DATA_IN_0_TENSOR_SIZE_DIM_0) begin
        assign image_done = 'b1;
        pixel_counter <= 0;
        // image_done becomes 1, 1 cycle delay after arith_valid, as when next_state is done,
        // pixel updates and checks in the same cycle. Then next cycles becomes 1 if check is 1
      end

      case (state)
        IDLE: begin
          image_done = 'b0;
          // Wait until both inputs are valid
          if (weight_valid && data_in_valid) begin
            // Capture the entire weight and data arrays.
            for (int i = 0; i < TOTAL_ELEMENTS; i++) begin
              weight_reg[i] <= weight_data[i];
              weight_reg_mul[i] = weight_data_mul[i*DATA_IN_0_PRECISION_0+:DATA_IN_0_PRECISION_0];
              data_reg[i] <= data_in[i];
            end
            // Initialize accumulator with the first multiplication result.
            accumulator <= $signed(weight_data[0]) * $signed(data_in[0]);
            accumulator_mul <= $signed(
                weight_data_mul[DATA_IN_0_PRECISION_0-1:0]
            ) * $signed(
                data_in[0]
            );
            counter <= 1;
            state <= ACCUM;
          end

        end

        ACCUM: begin
          // Process remaining elements one per clock cycle.
          if (pixel_counter == DATA_IN_0_TENSOR_SIZE_DIM_0 * DATA_IN_0_TENSOR_SIZE_DIM_0) begin
            state <= IDLE;
          end else if (counter < TOTAL_ELEMENTS) begin
            accumulator <= accumulator + ($signed(
                weight_reg[counter]
            ) * $signed(
                data_reg[counter]
            ));
            accumulator_mul <= accumulator_mul + ($signed(
                weight_reg_mul[counter]
            ) * $signed(
                data_reg[counter]
            ));
            counter <= counter + 1;
          end else begin
            state <= DONE; // when next_state is done-> i.e. one pixel finishes, hence pixel counter increments by 1
            pixel_counter <= pixel_counter + 1;
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
