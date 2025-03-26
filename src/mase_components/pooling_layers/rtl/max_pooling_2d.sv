`timescale 1ns / 1ps

module max_pooling_2d #(
    parameter DATA_IN_0_PRECISION_0  = 8,
    parameter DATA_IN_0_PRECISION_1  = 5,
    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 5,

    // Pooling parameters
    parameter POOL_SIZE = 2,
    parameter STRIDE    = 2,
    parameter PADDING   = 0,

    parameter DATA_IN_0_PARALLELISM_DIM_0 = 2,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 2,
    parameter DATA_IN_0_PARALLELISM_DIM_2 = 2,
    parameter DATA_IN_0_PARALLELISM_DIM_3 = 1,

    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_3 = 1,
    

   
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 2,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 2,
    parameter DATA_OUT_0_PARALLELISM_DIM_2 = 2,
    parameter DATA_OUT_0_PARALLELISM_DIM_3 = 1,

    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_3 = 1
    

    // FIFO related parameter (adjustable)
    
) (
    input logic clk,
    input logic rst,

    // Declare input data as signed
    input  logic signed [DATA_IN_0_PRECISION_0-1:0] data_in_0 [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1 * DATA_IN_0_PARALLELISM_DIM_2-1:0],
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    // Declare output data as signed
    output logic signed [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0 * DATA_OUT_0_PARALLELISM_DIM_1 * DATA_OUT_0_PARALLELISM_DIM_2-1:0],
    output logic data_out_0_valid,
    input logic data_out_0_ready
);

  // Set data width (e.g. 8 bits)
  localparam DATA_WIDTH = DATA_IN_0_PRECISION_0;
  localparam HEIGHT = DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1 * DATA_IN_0_PARALLELISM_DIM_2;
  localparam WIDTH  = HEIGHT;

  // For a 8x8 input and 2x2 pooling, but first acc 4 row to process
  localparam NUM_WINDOWS = HEIGHT; 
  localparam FIFO_DEPTH = 16;

  logic signed [DATA_IN_0_PRECISION_0 * HEIGHT-1:0] fifo_in_data_packed;
  logic signed [DATA_IN_0_PRECISION_0 * HEIGHT-1:0] fifo_out_data_packed;

  // assign fifo_in_data_packed = {data_in_0[0], data_in_0[1], data_in_0[2], data_in_0[3], data_in_0[4], data_in_0[5], data_in_0[6], data_in_0[7]};

  // wire signed [7:0] fifo_out_data_7 = fifo_out_data_packed[7:0];    
  // wire signed [7:0] fifo_out_data_6 = fifo_out_data_packed[15:8];   
  // wire signed [7:0] fifo_out_data_5 = fifo_out_data_packed[23:16];  
  // wire signed [7:0] fifo_out_data_4 = fifo_out_data_packed[31:24];  
  // wire signed [7:0] fifo_out_data_3 = fifo_out_data_packed[39:32];  
  // wire signed [7:0] fifo_out_data_2 = fifo_out_data_packed[47:40];  
  // wire signed [7:0] fifo_out_data_1 = fifo_out_data_packed[55:48];  
  // wire signed [7:0] fifo_out_data_0 = fifo_out_data_packed[63:56];

  always_comb begin
    fifo_in_data_packed = '0;
    for (int i = 0; i < HEIGHT; i++) begin
      fifo_in_data_packed[i * DATA_IN_0_PRECISION_0 +: DATA_IN_0_PRECISION_0] = data_in_0[i];
    end
  end

  logic signed [DATA_IN_0_PRECISION_0-1:0] fifo_out_data [0:HEIGHT-1];
  generate
    for (genvar i = 0; i < HEIGHT; i = i + 1) begin : gen_fifo_slices
      assign fifo_out_data[i] = fifo_out_data_packed[i * DATA_IN_0_PRECISION_0 +: DATA_IN_0_PRECISION_0];
    end
  endgenerate

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


  localparam WINDOW_ROWS = NUM_WINDOWS / 2;
  logic signed [DATA_WIDTH-1:0] row_buffer[0:WINDOW_ROWS-1][0:WIDTH-1];
  logic [$clog2(WINDOW_ROWS+1)-1:0] row_count;

  // =========================================================
  // 3) Define 4 pooling windows (each 2x2) with 4 elements each
  // Both window_regs and window_max are declared as signed
  logic signed [DATA_WIDTH-1:0] window_regs[0:NUM_WINDOWS-1][0:POOL_SIZE*POOL_SIZE-1];
  logic signed [DATA_WIDTH-1:0] window_max[0:NUM_WINDOWS-1];

  // Instantiate pool_window modules (each computes the maximum of 4 values)
  genvar k;
  generate
    for (k = 0; k < NUM_WINDOWS; k = k + 1) begin : gen_pw
      pool_window #(
          .DATA_WIDTH(DATA_WIDTH),
          .POOL_SIZE (POOL_SIZE)
      ) u_pool_window (
          .window_data(window_regs[k]),
          .max_value  (window_max[k])
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
  logic [$clog2(WINDOW_ROWS+1)-1:0] next_row_count;
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
        if (next_row_count == WINDOW_ROWS) next_state = PROCESS;
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
      for (r = 0; r < 8; r = r + 1) for (c = 0; c < 8; c = c + 1) row_buffer[r][c] <= 0;
    end else begin
      current_state    <= next_state;
      row_count        <= next_row_count;
      data_out_0_valid <= next_data_out_0_valid;
    end
  end


  // assign fifo_out_ready = next_fifo_out_ready;
  // always_ff @(posedge clk) begin
  //   if (!rst) begin
  //     if (fifo_out_ready && fifo_out_valid) begin
  //       for (int k = 0; k < WIDTH; k++) begin
  //         case (k)
  //           0: row_buffer[row_count][0] <= fifo_out_data_0;
  //           1: row_buffer[row_count][1] <= fifo_out_data_1;
  //           2: row_buffer[row_count][2] <= fifo_out_data_2;
  //           3: row_buffer[row_count][3] <= fifo_out_data_3;
  //           4: row_buffer[row_count][4] <= fifo_out_data_4;
  //           5: row_buffer[row_count][5] <= fifo_out_data_5;
  //           6: row_buffer[row_count][6] <= fifo_out_data_6;
  //           7: row_buffer[row_count][7] <= fifo_out_data_7;
  //         endcase
  //       end
  //     end
  //   end
  // end

  assign fifo_out_ready = next_fifo_out_ready;

  always_ff @(posedge clk) begin
    if (!rst) begin
      if (fifo_out_ready && fifo_out_valid) begin
        for (int k = 0; k < HEIGHT; k++) begin
          row_buffer[row_count][k] <= fifo_out_data[k];
        end
      end
    end
  end


  always_ff @(posedge clk) begin
    if (current_state == PROCESS) begin
      for (int row = 0; row < WINDOW_ROWS; row++) begin
        for (int col = 0; col < WINDOW_ROWS; col++) begin
          window_regs[row*2][col] <= row_buffer[row][col];
        end
        for (int col = 0; col < WINDOW_ROWS; col++) begin
          window_regs[row*2+1][col] <= row_buffer[row][col+4];
        end
      end
    end
  end


always_ff @(posedge clk) begin
  if (!rst && current_state == OUTPUT) begin
    for (int i = 0; i < WINDOW_ROWS; i++) begin
    data_out_0[i]            <= window_max[(WINDOW_ROWS-1-i)*2];
    data_out_0[i+WINDOW_ROWS] <= window_max[(WINDOW_ROWS-1-i)*2+1];
  end
  end
end


endmodule