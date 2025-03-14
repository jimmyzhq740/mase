`timescale 1ns / 1ps

module striding #(
    parameter DATA_WIDTH                  = 8,  // Bit width of pixel data
    parameter KERNEL_X                    = 3,  // Window width
    parameter KERNEL_Y                    = 3,  // Window height
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 4,  //Image width in x
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 4,
    parameter PADDING                     = 1
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic [DATA_WIDTH-1:0] pixel_in,
    input  logic                  pixel_in_valid,
    output logic                  pixel_in_ready,
    output logic [DATA_WIDTH-1:0] result_out          [KERNEL_X*KERNEL_Y-1:0],
    output logic                  sliding_window_valid,
    input  logic                  sliding_window_ready
);

  //--------------------------------------------------------------------------
  // Local Parameters and Type Declarations
  //--------------------------------------------------------------------------
  localparam COLS = DATA_IN_0_TENSOR_SIZE_DIM_0 + 2 * PADDING;  // Image width (including padding)
  localparam ROWS = DATA_IN_0_TENSOR_SIZE_DIM_1 + 2 * PADDING;
  localparam TOTAL_PIXELS = COLS * ROWS;
  localparam TOTAL_WINDOWS = (COLS - KERNEL_X + 1) * (ROWS - KERNEL_Y + 1);  // should be 16



  typedef enum logic [1:0] {
    COLLECT,
    OUTPUT
  } state_t;
  state_t state;


  (* keep = "true", verilator public_flat_rd *)logic [DATA_WIDTH-1:0] image[0:ROWS-1][0:COLS-1];

  // Pixel counter used during collection.
  logic [$clog2(TOTAL_PIXELS):0] pixel_count;
  logic pixel_valid_reg;


  // These registers define the top-left of the current window.
  logic [$clog2(ROWS - KERNEL_Y + 1)-1:0] win_row;
  logic [$clog2(COLS - KERNEL_X + 1)-1:0] win_col;

  // Counter for number of windows produced.
  logic [$clog2(TOTAL_WINDOWS):0] window_count;

  // Register to hold the extracted window.
  logic [DATA_WIDTH-1:0] window_reg[KERNEL_X*KERNEL_Y-1:0];

  // Drive the output from the register.
  genvar idx;
  generate
    for (idx = 0; idx < KERNEL_X * KERNEL_Y; idx = idx + 1) begin : assign_output
      assign result_out[idx] = window_reg[idx];
    end
  endgenerate


  // Accept new pixels only during COLLECT phase.
  assign pixel_in_ready = (state == COLLECT);


  logic sliding_window_valid_reg;
  assign sliding_window_valid = sliding_window_valid_reg;

  logic collect;
  logic output_s;

  initial begin
    collect  = 0;
    output_s = 0;
  end
  integer i, j;
  always_ff @(posedge clk or negedge rst_n) begin
    if (rst_n) begin
      state                    <= COLLECT;
      pixel_count              <= 0;
      window_count             <= 0;
      win_row                  <= 0;
      win_col                  <= 0;
      sliding_window_valid_reg <= 0;
      // Initialize image memory to 0.
      for (i = 0; i < ROWS; i = i + 1) for (j = 0; j < COLS; j = j + 1) image[i][j] <= 0;
      //$display("what is image: ",image[i][j]);
      // Clear window_reg.
      for (i = 0; i < KERNEL_X * KERNEL_Y; i = i + 1) window_reg[i] <= 0;
    end else begin
      // Default: deassert valid output.
      sliding_window_valid_reg <= 0;

      case (state)
        COLLECT: begin
          collect  = 1;
          output_s = 0;
          //pixel_valid_reg <= pixel_in_valid;
          //
          if (pixel_in_valid) begin
            // image [0][0]=0; image [0][1]=0; image[0][2]=0; image[0][3]=0; image[0][4]=0; image[0][5]=0;
            // image [1][0]=0; image [1][1]=1; image[1][2]=2; image[1][3]=3; image[1][4]=4; image[1][5]=0;
            // image [2][0]=0; image [2][1]=5; image[2][2]=6; image[2][3]=7; image[2][4]=8; image[2][5]=0;
            // image [3][0]=0; image [3][1]=9; image[3][2]=10; image[3][3]=11; image[3][4]=12; image[3][5]=0;
            // image [4][0]=0; image [4][1]=13; image[4][2]=14; image[4][3]=15; image[4][4]=16; image[4][5]=0;
            // image [5][0]=0; image [5][1]=0; image[5][2]=0; image[5][3]=0; image[5][4]=0; image[5][5]=0;
            //
            int r, c;
            r = pixel_count / COLS;
            c = pixel_count % COLS;
            image[r][c] <= pixel_in;
            // $display(" what is r: ", r, "  what is c: ", c);
            // $display(" what is image: ", pixel_in);
            pixel_count <= pixel_count + 1;
          end
          //
          if (pixel_count == TOTAL_PIXELS) state <= OUTPUT;
        end

        OUTPUT: begin

          output_s = 1;
          collect  = 0;

          // In the output phase, if downstream is ready and we haven't produced all windows,
          // load the window from the image memory.
          if (sliding_window_ready && (window_count < TOTAL_WINDOWS)) begin
            for (i = 0; i < KERNEL_Y; i = i + 1) begin
              for (j = 0; j < KERNEL_X; j = j + 1) begin
                // win_row = y
                // win_col = x
                window_reg[KERNEL_X*KERNEL_Y-1-(i*KERNEL_X+j)] <= image[win_row+i][win_col+j];
                // $display(" what is i: ", i, "  what is j: ", j);
                // $display(" what is win_row :", win_row, "  what is win_col:", win_col);
                // $display(" what is value: ", window_reg[i*KERNEL_X+j]);
              end
            end
            // Pulse valid for one clock cycle.
            sliding_window_valid_reg <= 1;
            window_count <= window_count + 1;
            // Update window pointer:
            if (win_col < (COLS - KERNEL_X)) win_col <= win_col + 1;
            else begin
              if (win_row == 1) begin
                win_col <= 1;
                if (win_row < (ROWS - KERNEL_Y)) win_row <= win_row + 1;
                else win_row <= win_row;  // remain if finished
              end

              win_col <= 0;
              if (win_row < (ROWS - KERNEL_Y)) win_row <= win_row + 1;
              else win_row <= win_row;  // remain if finished
            end
          end
          // Remain in OUTPUT state.
          state <= OUTPUT;
        end

        default: state <= COLLECT;
      endcase
    end
  end

endmodule
