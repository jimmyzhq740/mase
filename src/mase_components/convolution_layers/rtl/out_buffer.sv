`timescale 1ns / 1ps

module out_buffer #(
    parameter DATA_WIDTH                  = 8,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 4
) (
    input logic clk,
    input logic rst_n,

    input logic [DATA_WIDTH-1:0] data_in,
    input logic data_in_valid,
    output logic data_in_ready,
    output logic [DATA_WIDTH-1:0] data_out [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_valid,
    input logic data_out_ready
);

  // FULL: total number of rows (and columns) in the full matrix.
  // FULL = DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1.
  localparam integer FULL = DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1;
  // WIN: window dimension, set to DATA_IN_0_PARALLELISM_DIM_0 (e.g., 4)
  localparam integer WIN = DATA_IN_0_PARALLELISM_DIM_0;
  // Number of non-overlapping windows along rows and columns
  localparam integer NUM_WIN_ROWS = FULL / WIN;
  localparam integer NUM_WIN_COLS = FULL / WIN;

  // One buffer is used for accumulating new input (write side) and the other for window extraction (read side).

  logic [DATA_WIDTH-1:0] matrix_buf[0:1][0:FULL-1][0:FULL-1];


  //   - col_counter_w: counts elements in the current row (0 to FULL-1)
  //   - row_counter_w: counts completed rows in the current matrix (0 to FULL-1)
  reg [$clog2(FULL)-1:0] col_counter_w = 0;
  reg [$clog2(FULL)-1:0] row_counter_w = 0;
  // current_write indicates which buffer (0 or 1) is used for accumulation.
  reg current_write = 0;
  // current_read indicates which buffer is used for extraction.
  reg current_read = 1;
  // extraction_valid is asserted when the full matrix in the read buffer is ready for extraction.
  reg extraction_valid = 0;

  // The write side is ready as long as we have not filled the current matrix.
  // (If we are about to finish and the extraction is still active, back-pressure is applied.)
  assign data_in_ready = ((row_counter_w < FULL-1) || (col_counter_w < FULL-1)) ? 1'b1 : (!extraction_valid);

  //debug
  logic [1:0] valid_count;
  logic [1:0] ready_count;


  // These counters traverse the full matrix by window
  reg [$clog2(NUM_WIN_ROWS)-1:0] win_row_idx = 0;
  reg [$clog2(NUM_WIN_COLS)-1:0] win_col_idx = 0;
  // window_hold ensures that each window is held on output for one clock cycle.
  reg window_hold = 0;

  // The output is valid when a full matrix is available for extraction.
  assign data_out_valid = extraction_valid;


  // Write-Side Process: Accumulate scalar inputs into the write buffer.
  integer i;
  always_ff @(posedge clk or negedge rst_n) begin
    // if (rst_n) begin
    //   col_counter_w   <= 0;
    //   row_counter_w   <= 0;
    //   current_write   <= 0;
    //   current_read    <= 1;
    //   extraction_valid<= 0;
    // end else begin
    if (data_in_valid && data_in_ready) begin
      // Store the incoming scalar into the proper location.
      matrix_buf[current_write][row_counter_w][col_counter_w] <= data_in;
      // Check if we have reached the end of the current row.
      if (col_counter_w == FULL - 1) begin
        col_counter_w <= 0;
        // One row is complete.
        if (row_counter_w == FULL - 1) begin
          // The full matrix is complete.
          if (!extraction_valid) begin
            extraction_valid <= 1;  // Mark that the read buffer is valid.
            current_read <= current_write;  // Swap: the filled buffer becomes the read buffer.
            current_write <= ~current_write;  // The other buffer is now used for new inputs.
            row_counter_w <= 0;  // Reset the row counter for the new matrix.
          end
          // if extraction is still active, stall
        end else begin
          row_counter_w <= row_counter_w + 1;
        end
      end else begin
        // Otherwise, continue filling the current row.
        col_counter_w <= col_counter_w + 1;
      end
    end
  end
  //end

  always_ff @(posedge clk) begin
    if (data_in_valid) begin
      valid_count <= valid_count + 1;
    end
    if (data_in_ready) begin
      ready_count <= ready_count + 1;
    end
  end


  // Read-Side Process: Extract windows from the read buffer.
  always_ff @(posedge clk or negedge rst_n) begin
    if (extraction_valid) begin
      if (data_out_valid && data_out_ready) begin

        // Update window traversal counters.
        if (win_col_idx < NUM_WIN_COLS - 1) win_col_idx <= win_col_idx + 1;
        else begin
          win_col_idx <= 0;
          if (win_row_idx < NUM_WIN_ROWS - 1) win_row_idx <= win_row_idx + 1;
          else begin
            // All windows have been output; reset the window counters and invalidate extraction.
            win_row_idx <= 0;
            extraction_valid <= 0;
          end
        end
      end
    end
  end



  //   data_out[r*WIN + c] = matrix_buf[current_read][(win_row_idx*WIN)+r][(win_col_idx*WIN)+c]

  always_comb begin
    // Clear output by default.
    for (int idx = 0; idx < FULL; idx = idx + 1) data_out[idx] = '0;
    if (extraction_valid) begin
      for (int r = 0; r < WIN; r = r + 1) begin
        for (int c = 0; c < WIN; c = c + 1) begin
          data_out[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1-(r*WIN + c)] =
            matrix_buf[current_read][(win_row_idx * WIN) + r][(win_col_idx * WIN) + c];
          if (col_counter_w == row_counter_w == FULL - 1) begin
            //extraction_valid = 0;
            data_out[DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0] = 0;
          end
        end
      end
    end
  end

endmodule
