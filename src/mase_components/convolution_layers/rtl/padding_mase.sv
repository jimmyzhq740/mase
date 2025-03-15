`timescale 1ns / 1ps
module padding_mase #(
    parameter UNROLL = 1,
    parameter PADDING = 1,
    parameter DATA_IN_PRECISION_0 = 8,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 4,  //Image width in x
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 4  //Image width in y

) (
    input                                  clk,
    input                                  rst,
    input  logic [DATA_IN_PRECISION_0-1:0] data_in,
    input  logic                           data_in_valid,
    output logic                           data_in_ready,

    output logic [DATA_IN_PRECISION_0-1:0] data_out,
    output logic                           data_out_valid,
    input  logic                           data_out_ready
);

  // a common way to “pause” or “stall” a state machine until
  // the downstream consumer is ready is to look at the handshake signals (data_out_valid and data_out_ready) together.
  // If your design is producing valid data (data_out_valid = 1) but the consumer is not ready (data_out_ready = 0),
  // you simply don’t advance your internal counters or state machine.
  // In other words, hold your state and signals until data_out_ready goes high again.

  // Define the original coordinate for input data
  // Size: maximum counter number is DATA_IN_0_TENSOR_SIZE_DIM_0-1
  //If IMG_WIDTH = 6, then WIDTH_COUNT_BITS = $clog2(6), which is 3 bits for counter (because 2^3=8≥6).
  localparam X_ORIGINAL_COUNT_BITS = $clog2(DATA_IN_0_TENSOR_SIZE_DIM_0);
  localparam Y_ORIGINAL_COUNT_BITS = $clog2(DATA_IN_0_TENSOR_SIZE_DIM_1);
  logic [X_ORIGINAL_COUNT_BITS:0] x_origin_count;
  logic [Y_ORIGINAL_COUNT_BITS:0] y_origin_count;
  logic [X_ORIGINAL_COUNT_BITS:0] x_padding_count;
  logic [Y_ORIGINAL_COUNT_BITS:0] y_padding_count;

  // The width and height of the padded matrix
  localparam PADDED_W = DATA_IN_0_TENSOR_SIZE_DIM_0 + 2 * PADDING;


  initial begin
    assign x_origin_count = 'd1;
    assign y_origin_count = 'd1;
    assign x_padding_count = 'd0;
    assign y_padding_count = 'd0;
  end

  // I need a state_machine and a buffer:
  // States
  typedef enum logic [2:0] {
    S_IDLE,
    STATE1,
    STATE2,  //after
    STATE3,
    STATE4
  } state_t;

  state_t current_state, next_state;

  // A simple register to buffer the 1 input pixel
  logic [DATA_IN_PRECISION_0-1:0] stored_pixel;

  // State register
  always_ff @(posedge clk) begin
    if (rst) begin
      current_state   <= S_IDLE;
      x_padding_count <= 'd0;
      y_padding_count <= 'd0;
      x_origin_count  <= 'd1;
      y_origin_count  <= 'd1;
    end else begin
      current_state <= next_state;
    end

    if (current_state == S_IDLE) begin
      x_origin_count  <= 'd1;
      y_origin_count  <= 'd1;
      x_padding_count <= 'd0;
      y_padding_count <= 'd0;
      if (data_in_valid && data_in_ready) begin
        stored_pixel <= data_in;
      end
    end

    if (data_out_valid && !data_out_ready) begin
      x_padding_count <= x_padding_count;
      y_padding_count <= y_padding_count;
      x_origin_count  <= x_origin_count;
      y_origin_count  <= y_origin_count;
    end else if (data_out_valid && data_out_ready) begin
      // Capture the input pixel in S_IDLE
      // if (current_state == S_IDLE) begin
      //   x_origin_count  <= 'd1;
      //   y_origin_count  <= 'd1;
      //   x_padding_count <= 'd0;
      //   y_padding_count <= 'd0;
      //   if (data_in_valid && data_in_ready) begin
      //     stored_pixel <= data_in;
      //   end
      //end

      if (current_state == STATE1) begin
        if (x_padding_count == PADDED_W - 1) begin
          x_padding_count <= 0;
          y_padding_count <= y_padding_count + 1;
        end else begin
          x_padding_count <= x_padding_count + 1;
        end
      end

      if (current_state == STATE2) begin
        if (x_padding_count == PADDED_W - 1) begin
          x_padding_count <= 0;
          y_padding_count <= y_padding_count + 1;
        end else begin
          x_padding_count <= x_padding_count + 1;
        end
        //x_origin and y_origin keep increasing
        if (x_origin_count == DATA_IN_0_TENSOR_SIZE_DIM_0) begin  //4
          x_origin_count <= 'd1;
          if (y_origin_count == DATA_IN_0_TENSOR_SIZE_DIM_1) begin
            y_origin_count <= 'd1;
          end else begin
            y_origin_count <= y_origin_count + 1;
          end
        end else begin
          x_origin_count <= x_origin_count + 1;
        end
        stored_pixel <= data_in;
      end

      if (current_state == STATE3) begin
        if (x_padding_count == PADDED_W - 1) begin
          x_padding_count <= 0;
          y_padding_count <= y_padding_count + 1;
        end else begin
          x_padding_count <= x_padding_count + 1;
        end
        x_origin_count <= x_origin_count;
        y_origin_count <= y_origin_count;
        //x_origin and y_origin stop increasing
      end

      if (current_state == STATE4) begin
        if (x_padding_count == PADDED_W - 1) begin
          x_padding_count <= 0;
          if (y_padding_count == PADDED_W - 1) begin
            y_padding_count <= 0;
          end else begin
            y_padding_count <= y_padding_count + 1;
          end
        end else begin
          x_padding_count <= x_padding_count + 1;
        end
        x_origin_count <= x_origin_count;
        y_origin_count <= y_origin_count;
        //x_origin and y_origin stop increasing
      end
    end
  end

  //debug to check state I am in
  logic IDLE;
  logic state1;
  logic state2;
  logic state3;
  logic state4;



  // Next-state logic + output logic
  always_comb begin
    // Defaults
    data_in_ready  = 0;
    data_out_valid = 0;
    data_out       = 0;
    IDLE           = 0;
    state1         = 0;
    state2         = 0;
    state3         = 0;
    state4         = 0;
    next_state     = current_state;

    case (current_state)
      S_IDLE: begin
        // We haven't received any pixel yet
        // current state IDLE=1
        IDLE = 'd1;
        data_in_ready = 1;  // We can accept a pixel
        data_out_valid = 0;
        if (!data_out_ready) begin
          next_state = S_IDLE;
        end else begin
          if (data_in_valid && data_in_ready) begin
            // We just got our 1 input pixel => store it
            next_state = STATE1;  // Move to padded-output generation
            // assign IDLE = 'd0;
          end else begin
            next_state = S_IDLE;
          end
        end
      end

      STATE1: begin
        // We are producing the 6×6 padded frame, one pixel each cycle
        // STATE1: padding state
        data_out_valid = 1;  // We have a new output pixel every clock
        assign data_in_ready = 0;  // cannot let the reshape to send the signal
        assign state1 = 'd1;


        if (data_out_valid && !data_out_ready) begin
          // Stall here
          next_state = STATE1;
        end else if (data_out_valid && data_out_ready) begin
          // If we are at (x_pad,y_pad) = (1,1), output the stored real pixel
          // when x_pad, y_pad = (0,1)
          if (x_padding_count == 0 && y_padding_count == 1) begin
            // still outputting data_out =0 until x_pad, y_pad = (1,1)
            data_out   = '0;
            next_state = STATE2;
          end else begin
            data_out   = '0;  // pad
            next_state = STATE1;
          end
        end
      end

      // so in state2, we need to update the x_origin and y_origin as well, as we are sweeping the real data
      // the storage pixel needs to update the
      // state 2 origin number state
      STATE2: begin
        state2 = 'd1;
        data_out = stored_pixel;
        data_in_ready = 1;
        data_out_valid = 1;
        if (data_out_valid && !data_out_ready) begin
          // Stall here
          next_state = STATE2;
        end else if (data_out_valid && data_out_ready) begin
          if ((x_origin_count == DATA_IN_0_TENSOR_SIZE_DIM_0)) begin
            if (y_origin_count == DATA_IN_0_TENSOR_SIZE_DIM_1) begin
              // final state, where we try to output the last line of zeros
              next_state = STATE4;
            end else begin
              // output two zeros next state (next cycle)
              next_state = STATE3;
            end
          end else begin
            next_state = STATE2;
          end
        end
      end

      // after reaching the end of the original line, pause and wait for padding 0s out
      STATE3: begin
        state3 = 'd1;
        data_out = '0;
        data_in_ready = 0;
        data_out_valid = 1;
        if (data_out_valid && !data_out_ready) begin
          // Stall here
          next_state = STATE3;
        end else if (data_out_valid && data_out_ready) begin
          // when x_padding_count = 0 again, next state becomes 2, where we keep outputting original data
          if (x_padding_count == 0) begin
            next_state = STATE2;
          end else if ((x_padding_count == PADDED_W && y_padding_count == PADDED_W)) begin
            // last state where we output +zero second last line +last line of zero
            next_state = S_IDLE;
          end
        end
      end

      // outputing final line of zeros
      STATE4: begin
        state4 = 'd1;
        data_out = 'd0;
        data_in_ready = 1;
        data_out_valid = 1;
        if (data_out_valid && !data_out_ready) begin
          // Stall here
          next_state = STATE4;
        end else if (data_out_valid && data_out_ready) begin
          if ((x_padding_count == PADDED_W - 1 && y_padding_count == PADDED_W - 1)) begin
            next_state = S_IDLE;
          end
        end
      end
      default: next_state = S_IDLE;
    endcase
  end
endmodule


// [[[[15, 24,  2,  4],
//   [ 9, 20, 15, 28],
//   [14, 20, 11, 12],
//   [ 0,  5,  9, 16]]]]

// [
//   [0,  0,  0,  0,  0,  0],     // top padded row
//   [0, 15, 24,  2,  4,  0],
//   [0,  9, 20, 15, 28,  0],
//   [0, 14, 20, 11, 12,  0],
//   [0,  0,  5,  9, 16,  0],
//   [0,  0,  0,  0,  0,  0]      // bottom padded row
// ]

