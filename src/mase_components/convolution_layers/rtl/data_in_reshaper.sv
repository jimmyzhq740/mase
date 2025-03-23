`timescale 1ns / 1ps

module data_in_reshaper #(
    parameter DATA_WIDTH      = 16,  // Data width
    parameter IMG_WIDTH       = 4,   // Image width
    parameter IMG_HEIGHT      = 4,   // Image height
    parameter IN_CHANNELS     = 4,   // Input channels
    parameter BATCH_SIZE      = 2,   // Number of batches
    parameter UNROLL_CHANNELS = 2,   // Number of channels processed in parallel
    parameter SPATIAL_GROUP_X = 2,   // X dimension of spatial groups
    parameter SPATIAL_GROUP_Y = 2    // Y dimension of spatial groups
) (
    input logic clk,
    input logic rst,

    // Flattened input data array (new organization)
    input  logic [DATA_WIDTH-1:0] data_in[BATCH_SIZE*UNROLL_CHANNELS*SPATIAL_GROUP_Y*SPATIAL_GROUP_X-1:0],
    input logic data_in_valid,
    output logic data_in_ready,

    // Reshaped output data (UNROLL_CHANNELS per cycle)
    output logic [DATA_WIDTH-1:0] data_out[UNROLL_CHANNELS-1:0],
    output logic data_out_valid,
    input logic data_out_ready
);

  localparam TOTAL_PIXELS = IMG_WIDTH * IMG_HEIGHT;
  localparam TOTAL_GROUPS = (IN_CHANNELS + UNROLL_CHANNELS - 1) / UNROLL_CHANNELS;
  localparam SPATIAL_GROUPS_X = IMG_WIDTH / SPATIAL_GROUP_X;
  localparam SPATIAL_GROUPS_Y = IMG_HEIGHT / SPATIAL_GROUP_Y;
  localparam INPUT_ARRAY_SIZE = BATCH_SIZE * UNROLL_CHANNELS * SPATIAL_GROUP_Y * SPATIAL_GROUP_X;
  localparam BUFFER_SIZE = BATCH_SIZE * IN_CHANNELS * IMG_HEIGHT * IMG_WIDTH;

  // Input buffer to store received data
  logic [DATA_WIDTH-1:0] data_buffer[BUFFER_SIZE-1:0];

  // Buffer validity tracking - NEW
  logic buffer_valid[BUFFER_SIZE-1:0];

  // Position counters
  logic [$clog2(BATCH_SIZE):0] batch_count;
  logic [$clog2(TOTAL_GROUPS):0] channel_group;
  logic [$clog2(IMG_WIDTH):0] x_count;
  logic [$clog2(IMG_HEIGHT):0] y_count;
  logic processing;

  // Input counters for continuous filling
  logic [$clog2(SPATIAL_GROUPS_X):0] spatial_group_x_count;
  logic [$clog2(SPATIAL_GROUPS_Y):0] spatial_group_y_count;
  logic [$clog2(TOTAL_GROUPS):0] input_channel_group;
  logic [$clog2(BATCH_SIZE):0] input_batch_count;

  // Status tracking
  logic next_batch_ready;  // Indicates when ready for next input batch

  // Debug counters
  int input_count;
  int output_count;

  // Output position tracking - NEW
  int current_output_idx;
  int next_output_idx;

  // Complete processing tracking - NEW
  logic all_data_processed;
  int pixels_processed;
  int total_pixels_to_process;

  // Initialize buffer validity - NEW
  initial begin
    for (int i = 0; i < BUFFER_SIZE; i++) begin
      buffer_valid[i] = 0;
    end
    all_data_processed = 0;
    pixels_processed = 0;
    total_pixels_to_process = BATCH_SIZE * IN_CHANNELS * IMG_HEIGHT * IMG_WIDTH;
  end

  // Current output index computation
  always_comb begin
    current_output_idx = batch_count * IN_CHANNELS * IMG_HEIGHT * IMG_WIDTH +
                       (channel_group * UNROLL_CHANNELS) * IMG_HEIGHT * IMG_WIDTH +
                       y_count * IMG_WIDTH + x_count;

    // Calculate next output position index with wrapping logic
    if (x_count == IMG_WIDTH - 1) begin
      if (y_count == IMG_HEIGHT - 1) begin
        if (channel_group == TOTAL_GROUPS - 1) begin
          if (batch_count == BATCH_SIZE - 1) begin
            // Last pixel in last batch
            next_output_idx = 0;  // Wrap to beginning
          end else begin
            // First pixel in next batch
            next_output_idx = (batch_count + 1) * IN_CHANNELS * IMG_HEIGHT * IMG_WIDTH;
          end
        end else begin
          // First pixel in next channel group
          next_output_idx = batch_count * IN_CHANNELS * IMG_HEIGHT * IMG_WIDTH +
                           ((channel_group + 1) * UNROLL_CHANNELS) * IMG_HEIGHT * IMG_WIDTH;
        end
      end else begin
        // First pixel in next row
        next_output_idx = batch_count * IN_CHANNELS * IMG_HEIGHT * IMG_WIDTH +
                         (channel_group * UNROLL_CHANNELS) * IMG_HEIGHT * IMG_WIDTH +
                         (y_count + 1) * IMG_WIDTH;
      end
    end else begin
      // Next pixel in same row
      next_output_idx = current_output_idx + 1;
    end
  end

  // Buffer filling state machine - WITH REVERSED ELEMENT ORDER
  always_ff @(posedge clk) begin
    if (rst) begin
      spatial_group_x_count <= 0;
      spatial_group_y_count <= 0;
      input_channel_group <= 0;
      input_batch_count <= 0;
      input_count <= 0;
      next_batch_ready <= 1;  // Ready to accept first batch after reset

      // Clear buffer validity
      for (int i = 0; i < BUFFER_SIZE; i++) begin
        buffer_valid[i] = 0;
      end

    end else if (data_in_valid && data_in_ready) begin
      // Increment counter whenever we receive a batch
      input_count <= input_count + 1;

      // Store data in buffer according to new input organization
      // with reversed element order
      for (int b = 0; b < BATCH_SIZE; b++) begin
        for (int c = 0; c < UNROLL_CHANNELS; c++) begin
          for (int y = 0; y < SPATIAL_GROUP_Y; y++) begin
            for (int x = 0; x < SPATIAL_GROUP_X; x++) begin
              // REVERSED: Calculate input index from end of array
              automatic
              int
              input_idx = INPUT_ARRAY_SIZE - 1 - (
                                                    b * UNROLL_CHANNELS * SPATIAL_GROUP_Y * SPATIAL_GROUP_X +
                                                    c * SPATIAL_GROUP_Y * SPATIAL_GROUP_X +
                                                    y * SPATIAL_GROUP_X + x);

              // Normal buffer index calculation
              automatic
              int
              buffer_idx = (input_batch_count + b) % BATCH_SIZE * IN_CHANNELS * IMG_HEIGHT * IMG_WIDTH +
                                                     (input_channel_group * UNROLL_CHANNELS + c) * IMG_HEIGHT * IMG_WIDTH +
                                                     (spatial_group_y_count * SPATIAL_GROUP_Y + y) * IMG_WIDTH +
                                                     (spatial_group_x_count * SPATIAL_GROUP_X + x);

              // Check that we're not exceeding buffer bounds
              if (buffer_idx < BUFFER_SIZE && input_idx < INPUT_ARRAY_SIZE) begin
                data_buffer[buffer_idx]  = data_in[input_idx];
                buffer_valid[buffer_idx] = 1;  // Mark this location as valid
              end
            end
          end
        end
      end

      // Update input group counters
      if (spatial_group_x_count >= SPATIAL_GROUPS_X - 1) begin
        spatial_group_x_count <= 0;
        if (spatial_group_y_count >= SPATIAL_GROUPS_Y - 1) begin
          spatial_group_y_count <= 0;
          if (input_channel_group >= TOTAL_GROUPS - 1) begin
            input_channel_group <= 0;

            // Move to the next batch
            if (input_batch_count >= BATCH_SIZE - 1) begin
              input_batch_count <= 0;
            end else begin
              input_batch_count <= input_batch_count + 1;
            end

            next_batch_ready <= 1;  // Ready for next batch immediately
          end else begin
            input_channel_group <= input_channel_group + 1;
            next_batch_ready <= 1;  // Ready for next batch immediately
          end
        end else begin
          spatial_group_y_count <= spatial_group_y_count + 1;
          next_batch_ready <= 1;  // Ready for next batch immediately
        end
      end else begin
        spatial_group_x_count <= spatial_group_x_count + 1;
        next_batch_ready <= 1;  // Ready for next batch immediately
      end
    end
  end

  // Output processing state machine - MODIFIED to check data availability
  always_ff @(posedge clk) begin
    if (rst) begin
      batch_count <= 0;
      channel_group <= 0;
      x_count <= 0;
      y_count <= 0;
      processing <= 0;
      data_out_valid <= 0;
      output_count <= 0;
      pixels_processed <= 0;
      all_data_processed <= 0;
    end else if (!processing && buffer_valid[current_output_idx]) begin
      // Start/resume processing when data at the current output position is available
      processing <= 1;
      data_out_valid <= 1;
    end else if (processing && data_out_ready) begin
      output_count <= output_count + 1;
      pixels_processed <= pixels_processed + 1;

      if (pixels_processed + 1 >= total_pixels_to_process) begin
        all_data_processed <= 1;
      end

      // Mark this data as processed (optional, can be removed to save logic)
      buffer_valid[current_output_idx] <= 0;

      // Update position counters
      if (x_count >= IMG_WIDTH - 1) begin
        x_count <= 0;
        if (y_count >= IMG_HEIGHT - 1) begin
          y_count <= 0;
          if (channel_group >= TOTAL_GROUPS - 1) begin
            channel_group <= 0;
            if (batch_count >= BATCH_SIZE - 1) begin
              batch_count <= 0;
              pixels_processed <= 0;  // Reset for next processing cycle
              all_data_processed <= 0;
            end else begin
              batch_count <= batch_count + 1;
            end
          end else begin
            channel_group <= channel_group + 1;
          end
        end else begin
          y_count <= y_count + 1;
        end
      end else begin
        x_count <= x_count + 1;
      end

      // Check if the next output position data is available
      if (buffer_valid[next_output_idx]) begin
        data_out_valid <= 1;
      end else begin
        data_out_valid <= 0;  // Pause output until the next data is available
        processing <= 0;     // Pause processing
      end
    end
  end

  // Data output generation
  genvar i;
  generate
    for (i = 0; i < UNROLL_CHANNELS; i++) begin : gen_channel_select
      always_comb begin
        if (processing && buffer_valid[current_output_idx] &&
            (channel_group * UNROLL_CHANNELS + i < IN_CHANNELS)) begin
          // Calculate buffer index for output
          automatic
          int
          buffer_idx_2 = batch_count * IN_CHANNELS * IMG_HEIGHT * IMG_WIDTH +
                         (channel_group * UNROLL_CHANNELS + i) * IMG_HEIGHT * IMG_WIDTH +
                         y_count * IMG_WIDTH + x_count;

          if (buffer_idx_2 < BUFFER_SIZE) begin

            data_out[UNROLL_CHANNELS-1-i] = data_buffer[buffer_idx_2];
          end else begin
            data_out[UNROLL_CHANNELS-1-i] = 0;
          end
        end else begin
          data_out[UNROLL_CHANNELS-1-i] = 0;
        end
      end
    end
  endgenerate


  // Accept new data if we're not in the middle of processing the current output
  // or if we're processing but downstream module is ready for more data
  assign data_in_ready = next_batch_ready && (!all_data_processed);

endmodule
