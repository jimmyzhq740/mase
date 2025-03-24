`timescale 1ns / 1ps
`timescale 1ns / 1ps

module pool_window #(
    parameter DATA_WIDTH = 8,
    parameter POOL_SIZE  = 2
) (
    input  logic signed [DATA_WIDTH-1:0] window_data[POOL_SIZE*POOL_SIZE],
    output logic signed [DATA_WIDTH-1:0] max_value
) (
    input  logic signed [DATA_WIDTH-1:0] window_data[POOL_SIZE*POOL_SIZE],
    output logic signed [DATA_WIDTH-1:0] max_value
);
  always_comb begin
    max_value = window_data[0];
    for (int i = 1; i < POOL_SIZE * POOL_SIZE; i++) begin
      if (window_data[i] > max_value) max_value = window_data[i];
    end
  end
  always_comb begin
    max_value = window_data[0];
    for (int i = 1; i < POOL_SIZE * POOL_SIZE; i++) begin
      if (window_data[i] > max_value) max_value = window_data[i];
    end
  end
endmodule
