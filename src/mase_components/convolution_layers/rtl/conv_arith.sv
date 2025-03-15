module conv_arith #(
    parameter DATA_IN_0_PRECISION_0   = 16,
    parameter WEIGHT_PRECISION_0      = 8,
    parameter WEIGHT_TENSOR_SIZE_DIM_0 = 3,
    parameter WEIGHT_TENSOR_SIZE_DIM_1 = 3,
    // ARITH_DATA_OUT_WIDTH = 16 + 8 + $clog2(3*3) = 16 + 8 + 4 = 28 bits
    parameter ARITH_DATA_OUT_WIDTH    = DATA_IN_0_PRECISION_0 + WEIGHT_PRECISION_0 + $clog2(WEIGHT_TENSOR_SIZE_DIM_0*WEIGHT_TENSOR_SIZE_DIM_1)
)(
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
   typedef enum logic [1:0] {IDLE, ACCUM, DONE} state_t;
   state_t state;

   // Internal registers to hold the input arrays
   logic signed [WEIGHT_PRECISION_0-1:0] weight_reg [0:TOTAL_ELEMENTS-1];
   logic signed [DATA_IN_0_PRECISION_0-1:0] data_reg   [0:TOTAL_ELEMENTS-1];
   logic signed [ARITH_DATA_OUT_WIDTH-1:0]   accumulator;


   // Counter for processing elements and accumulator for result
   // Counter width is enough to count up to TOTAL_ELEMENTS (9)
   logic [$clog2(TOTAL_ELEMENTS+1)-1:0] counter;


   // Output assignment: the arithmetic output reflects the accumulator value.
   assign arith_data_out = accumulator;

   // The FSM and sequential multiply-accumulate process.
   always_ff @(posedge clk or posedge rst) begin
      if (rst) begin
         state         <= IDLE;
         weight_ready  <= 1;
         data_in_ready <= 1;
         arith_valid   <= 0;
         counter       <= 0;
         accumulator   <= 0;
      end else begin
         case (state)
            IDLE: begin
               arith_valid <= 0;
               // When both inputs are valid, capture the entire weight and data arrays.
               if (weight_valid && data_in_valid) begin
                  for (int i = 0; i < TOTAL_ELEMENTS; i++) begin
                     weight_reg[i] <= weight_data[i];
                     data_reg[i]   <= data_in[i];
                  end
                  // Initialize accumulator with the first multiplication result.
                  accumulator <= $signed(weight_data[0]) * $signed(data_in[0]);
                  // Set counter to process the remaining elements.
                  counter     <= 1;
                  // Indicate that new inputs are not accepted until current computation completes.
                  weight_ready  <= 0;
                  data_in_ready <= 0;
                  state         <= ACCUM;
               end
            end
            ACCUM: begin
               // Process remaining elements one per clock cycle.
               if (counter < TOTAL_ELEMENTS) begin
                  accumulator <= accumulator + ($signed(weight_reg[counter]) * $signed(data_reg[counter]));

                  counter     <= counter + 1;
               end else begin
                  // When all elements have been processed, move to DONE state.
                  state <= DONE;
               end
            end
            DONE: begin
               // In DONE state, assert arith_valid to indicate the result is ready.
               arith_valid <= 1;
               // Wait for the handshake (arith_ready) to go back to IDLE.
               if (arith_ready) begin
                  weight_ready  <= 1;
                  data_in_ready <= 1;
                  arith_valid   <= 0;
                  state         <= IDLE;
               end
            end
            default: state <= IDLE;
         endcase
      end
   end

endmodule
