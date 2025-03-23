`timescale 1ns / 1ps

module max_pooling_2d #(
    // Data precision parameters (signed)
    parameter DATA_IN_0_PRECISION_0  = 8,
    parameter DATA_IN_0_PRECISION_1  = 5,
    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 5,

    // Pooling parameters (for a 4x4 input and 2x2 pooling)
    parameter POOL_SIZE = 2,
    parameter STRIDE    = 0,
    parameter PADDING   = 0,

    // Input interface: assumes 4 data per cycle (representing one full row)
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 2,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 2,

    // Unused parameters (for matching external interface with 4D tensor)
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_3 = 1,

    parameter DATA_IN_0_PARALLELISM_DIM_2 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_3 = 1,

    // Output interface: assumes output of 4 data (corresponding to 4 pooling blocks)
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 4,

    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_2 = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_3 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_3 = 1,

    // FIFO related parameter (adjustable)
    parameter FIFO_DEPTH = 4
) (
    input logic clk,
    input logic rst,

    // Declare input data as signed
    input  logic signed [DATA_IN_0_PRECISION_0-1:0] data_in_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    // Declare output data as signed
    output logic signed [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_0_valid,
    input logic data_out_0_ready
);

  // Set data width (e.g. 8 bits)
  localparam DATA_WIDTH = DATA_IN_0_PRECISION_0;
  localparam HEIGHT = POOL_SIZE * POOL_SIZE;
  localparam WIDTH = POOL_SIZE * POOL_SIZE;

  // For a 4x4 input and 2x2 pooling, number of windows is 4
  localparam NUM_WINDOWS = (WIDTH / POOL_SIZE) * (HEIGHT / POOL_SIZE);  // 4

  // Pack 4x8bit data into one 32-bit word to send to the FIFO
  // Note: data_in_0[3] is the MSB, data_in_0[0] is the LSB
  logic signed [DATA_IN_0_PRECISION_0*WIDTH-1:0] fifo_in_data_packed;
  logic signed [DATA_IN_0_PRECISION_0*WIDTH-1:0] fifo_out_data_packed;
  assign fifo_in_data_packed = {data_in_0[3], data_in_0[2], data_in_0[1], data_in_0[0]};

  // Unpack the FIFO output into 4 8-bit values
  wire signed [7:0] fifo_out_data_0 = fifo_out_data_packed[7:0];
  wire signed [7:0] fifo_out_data_1 = fifo_out_data_packed[15:8];
  wire signed [7:0] fifo_out_data_2 = fifo_out_data_packed[23:16];
  wire signed [7:0] fifo_out_data_3 = fifo_out_data_packed[31:24];

  // FIFO signals
  wire fifo_out_valid;
  wire fifo_out_ready;
  wire fifo_empty, fifo_full;
  fifo #(
      .DATA_WIDTH(DATA_IN_0_PRECISION_0 * WIDTH),
      .DEPTH(FIFO_DEPTH)
  ) u_fifo (
      .clk(clk),
      .rst(rst),
      .in_data(fifo_in_data_packed),
      .in_valid(data_in_0_valid),
      .in_ready(data_in_0_ready),
      .out_data(fifo_out_data_packed),
      .out_valid(fifo_out_valid),
      .out_ready(fifo_out_ready),
      .empty(fifo_empty),
      .full(fifo_full)
  );

  // =========================================================
  // 2) Accumulate 4 rows of data: use row_buffer to store 4 rows (each row has 4 elements)
  // Define row_buffer as signed
  logic signed [DATA_WIDTH-1:0] row_buffer[0:HEIGHT-1][0:WIDTH-1];
  logic [$clog2(HEIGHT+1)-1:0] row_count;

  // =========================================================
  // 3) Define 4 pooling windows (each 2x2) with 4 elements each
  // Both window_regs and window_max are declared as signed
  logic signed [DATA_WIDTH-1:0] window_regs[0:NUM_WINDOWS-1][0:POOL_SIZE*POOL_SIZE-1];
  logic signed [DATA_WIDTH-1:0] window_max[0:NUM_WINDOWS-1];

  // Instantiate pool_window modules (each computes the maximum of 4 values)
  genvar i;
  generate
    for (i = 0; i < NUM_WINDOWS; i = i + 1) begin : gen_pw
      pool_window #(
          .DATA_WIDTH(DATA_WIDTH),
          .POOL_SIZE (POOL_SIZE)
      ) u_pool_window (
          .window_data(window_regs[i]),
          .max_value  (window_max[i])
      );
    end
  endgenerate

  // State machine control: IDLE, BUFFER, PROCESS, OUTPUT
  typedef enum logic [1:0] {
    IDLE,
    BUFFER,
    PROCESS,
    OUTPUT
  } state_t;
  state_t current_state, next_state;

  // State machine control signals
  logic [$clog2(HEIGHT+1)-1:0] next_row_count;
  logic next_data_out_0_valid;
  logic next_fifo_out_ready;

  always_comb begin
    next_state = current_state;
    next_row_count = row_count;
    next_data_out_0_valid = 0;
    next_fifo_out_ready = 1'b0;
    case (current_state)
      IDLE: begin
        next_row_count = 0;
        if (fifo_out_valid) begin
          next_fifo_out_ready = 1;
          next_row_count = 1;
          next_state = BUFFER;
        end
      end
      BUFFER: begin
        if (row_count < HEIGHT) begin
          if (fifo_out_valid) begin
            next_fifo_out_ready = 1;
            next_row_count = row_count + 1;
          end
        end
        if (next_row_count == HEIGHT) next_state = PROCESS;
      end
      PROCESS: begin
        next_state = OUTPUT;
      end
      OUTPUT: begin
        next_data_out_0_valid = 1;
        if (data_out_0_ready) begin
          next_row_count = 0;
          next_state = IDLE;
        end
      end
      default: begin
        next_state = IDLE;
        next_row_count = 0;
        next_data_out_0_valid = 0;
      end
    endcase
  end

  integer r, c;
  always_ff @(posedge clk) begin
    if (rst) begin
      current_state    <= IDLE;
      row_count        <= 0;
      data_out_0_valid <= 0;
      // Initialize all row_buffer elements to 0
      for (r = 0; r < HEIGHT; r = r + 1) for (c = 0; c < WIDTH; c = c + 1) row_buffer[r][c] <= 0;
    end else begin
      current_state    <= next_state;
      row_count        <= next_row_count;
      data_out_0_valid <= next_data_out_0_valid;
    end
  end


  // Read FIFO data and store one row into row_buffer
  assign fifo_out_ready = next_fifo_out_ready;
  always_ff @(posedge clk) begin
    if (!rst) begin
      if (fifo_out_ready && fifo_out_valid) begin
        for (int k = 0; k < WIDTH; k++) begin
          case (k)
            0: row_buffer[row_count][0] <= fifo_out_data_0;
            1: row_buffer[row_count][1] <= fifo_out_data_1;
            2: row_buffer[row_count][2] <= fifo_out_data_2;
            3: row_buffer[row_count][3] <= fifo_out_data_3;
          endcase
        end
      end
    end
  end


  // PROCESS state row major
  always_ff @(posedge clk) begin
    if (current_state == PROCESS) begin
      for (int i = 0; i < NUM_WINDOWS; i++) begin
        for (int j = 0; j < NUM_WINDOWS; j++) begin
          window_regs[i][j] <= row_buffer[i][j];
        end
      end
    end
  end


  // OUTPUT state: Output the max values for each block
  always_ff @(posedge clk) begin
    if (!rst && current_state == OUTPUT) begin
      for (int i = 0; i < NUM_WINDOWS; i++) begin
        data_out_0[i] <= window_max[NUM_WINDOWS-1-i];
      end
    end
  end

endmodule
