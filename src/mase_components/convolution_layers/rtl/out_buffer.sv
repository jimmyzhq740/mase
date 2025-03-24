`timescale 1ns / 1ps

module out_buffer #(

    parameter DATA_WIDTH = 8,
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 4,  // Window height (and width)
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 4, // Used to compute the number of elements per row in the input stream
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 16,  // Full matrix row count
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 16  // Full matrix column count
) (
    input logic clk,
    input logic rst_n,

    input logic [DATA_WIDTH-1:0] data_in,
    input logic data_in_valid,
    output logic data_in_ready,
    output logic [DATA_WIDTH-1:0] data_out [DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1 - 1:0],
    output logic data_out_valid,
    input logic data_out_ready
);

  // TENSOR_ROWS: number of rows in the full tensor.
  localparam integer TENSOR_ROWS = DATA_IN_0_TENSOR_SIZE_DIM_0;  // e.g., 16
  // TENSOR_COLS: number of columns in the full tensor.
  localparam integer TENSOR_COLS = DATA_IN_0_TENSOR_SIZE_DIM_1;
  localparam integer WIN = DATA_IN_0_PARALLELISM_DIM_0;
  // Compute the number of windows horizontally and vertically.
  localparam integer NUM_WIN_H = TENSOR_COLS / WIN;
  localparam integer NUM_WIN_V = TENSOR_ROWS / WIN;


  logic [DATA_WIDTH-1:0] matrix_buf[0:1][0:TENSOR_ROWS-1][0:TENSOR_COLS-1];


  // col_cnt: counts the number of elements received in the current row (0 to TENSOR_COLS-1)
  reg [$clog2(TENSOR_COLS)-1:0] col_cnt;
  // row_cnt: counts the number of rows accumulated in the current tensor (0 to TENSOR_ROWS-1)
  reg [$clog2(TENSOR_ROWS)-1:0] row_cnt;
  // current_write: index (0 or 1) of the current buffer used for accumulating input data.
  reg current_write;
  // current_read: index of the buffer currently used for window extraction.
  reg current_read;
  // batch_valid: asserted when the current tensor (full matrix) is completely accumulated.
  reg batch_valid;


  // win_row_idx: vertical index for window extraction, ranges from 0 to NUM_WIN_V-1
  reg [$clog2(NUM_WIN_V)-1:0] win_row_idx;
  // win_col_idx: horizontal index for window extraction, ranges from 0 to NUM_WIN_H-1
  reg [$clog2(NUM_WIN_H)-1:0] win_col_idx;
  // window_hold: ensures that each window is output for at least one clock cycle.
  reg window_hold;

  // The output valid signal is directly driven by batch_valid.
  assign data_out_valid = batch_valid;

  assign data_in_ready = ((row_cnt < TENSOR_ROWS) || (col_cnt < TENSOR_COLS-1)) ? 1'b1 : (!batch_valid);


  // Write-Side Process: Accumulate scalar input data to form a full tensor.

  integer i;
  always_ff @(posedge clk or negedge rst_n) begin
    if (rst_n) begin
      col_cnt       <= 0;
      row_cnt       <= 0;
      current_write <= 0;
      current_read  <= 1;
      batch_valid   <= 0;
    end else begin
      if (data_in_valid && data_in_ready) begin
        // Store the incoming data element in the current write buffer.
        matrix_buf[current_write][row_cnt][col_cnt] <= data_in;
        // Check if the current row is complete.
        if (col_cnt == TENSOR_COLS - 1) begin
          col_cnt <= 0;
          // The current row is complete; check if the full tensor is complete.
          if (row_cnt == TENSOR_ROWS - 1) begin
            // The full tensor has been accumulated.
            if (!batch_valid) begin
              batch_valid   <= 1;  // Mark the tensor as valid for extraction.
              // Swap buffers: current write buffer becomes the read buffer.
              current_read  <= current_write;
              current_write <= ~current_write;
              row_cnt       <= 0;  // Reset the row counter for the new tensor.
            end
          end else begin
            row_cnt <= row_cnt + 1;
          end
        end else begin
          col_cnt <= col_cnt + 1;
        end
      end
    end
  end


  // Read-Side Process: Extract windows from the full tensor in the read buffer.

  always_ff @(posedge clk or negedge rst_n) begin
    if (rst_n) begin
      win_row_idx <= 0;
      win_col_idx <= 0;
      window_hold <= 0;
    end else if (batch_valid) begin
      if (data_out_valid && data_out_ready) begin

        // Update horizontal window index first.
        if (win_col_idx < NUM_WIN_H - 1) win_col_idx <= win_col_idx + 1;
        else begin
          win_col_idx <= 0;
          // Then update vertical window index.
          if (win_row_idx < NUM_WIN_V - 1) win_row_idx <= win_row_idx + 1;
          else begin
            // All windows in the full tensor have been output.
            win_row_idx <= 0;
            batch_valid <= 0;  // End output; wait for the next tensor.
          end
        end
      end
    end
  end



  //   data_out[(WIN*WIN - 1) - (i*WIN + j)] = matrix_buf[current_read][(win_row_idx*WIN)+i][(win_col_idx*WIN)+j];

  always_comb begin
    // Clear the output array by default.
    for (
        int idx = 0; idx < DATA_IN_0_PARALLELISM_DIM_0 * DATA_IN_0_PARALLELISM_DIM_1; idx = idx + 1
    )
      data_out[idx] = '0;
    if (batch_valid) begin
      for (int i = 0; i < WIN; i = i + 1) begin
        for (int j = 0; j < WIN; j = j + 1) begin
          data_out[(WIN*WIN - 1) - (i*WIN + j)] =
            matrix_buf[current_read][(win_row_idx * WIN) + i][(win_col_idx * WIN) + j];
        end
      end
    end
  end

endmodule
