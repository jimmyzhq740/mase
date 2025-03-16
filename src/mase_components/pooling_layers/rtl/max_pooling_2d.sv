`timescale 1ns / 1ps

module max_pooling_2d #(
    // 数据精度参数（有符号数）
    parameter DATA_IN_0_PRECISION_0  = 8,
    parameter DATA_IN_0_PRECISION_1  = 5,
    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 5,

    parameter WIDTH                       = 4,
    parameter HEIGHT                      = 4,
    parameter POOL_SIZE                   = 2,
    parameter STRIDE                      = 0,
    parameter PADDING                     = 0,
    // 输入接口：假设每个周期收到 4 个数据（代表一整行）
    parameter DATA_IN_0_PARALLELISM_DIM_0 = 2,
    parameter DATA_IN_0_PARALLELISM_DIM_1 = 2,
    // 新增未使用的参数（例如，为了匹配外部接口要求4D张量）
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_3 = 1,

    parameter DATA_IN_0_PARALLELISM_DIM_2  = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_3  = 1,
    // 输出接口：假设输出 4 个数据（对应 4 个池化 block）
    parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_1 = 4,
    // 同样新增未使用的输出参数
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_2 = 1,
    parameter DATA_OUT_0_TENSOR_SIZE_DIM_3 = 1,
    parameter DATA_OUT_0_PARALLELISM_DIM_3 = 1,

    // ========= FIFO 相关参数，根据需要可调 =========
    parameter FIFO_DEPTH = 4
) (
    input logic clk,
    input logic rst,

    // 输入数据声明为 signed
    input  logic signed [DATA_IN_0_PRECISION_0-1:0] data_in_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input logic data_in_0_valid,
    output logic data_in_0_ready,

    // 输出数据也设为 signed
    output logic signed [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_0_valid,
    input logic data_out_0_ready
);

  localparam DATA_WIDTH = DATA_IN_0_PRECISION_0;  // 8

  // 根据 4×4 输入和 2×2 池化，窗口数量为 4
  localparam NUM_WINDOWS = (WIDTH / POOL_SIZE) * (HEIGHT / POOL_SIZE);  // 4

  // =========================================================
  // 1) 将 4×8bit 数据打包送入 FIFO
  // 注意：这里将打包的数据也声明为 signed
  logic signed [DATA_IN_0_PRECISION_0*WIDTH-1:0] fifo_in_data_packed;
  logic signed [DATA_IN_0_PRECISION_0*WIDTH-1:0] fifo_out_data_packed;
  assign fifo_in_data_packed = {data_in_0[3], data_in_0[2], data_in_0[1], data_in_0[0]};

  wire signed [7:0] fifo_out_data_0 = fifo_out_data_packed[7:0];
  wire signed [7:0] fifo_out_data_1 = fifo_out_data_packed[15:8];
  wire signed [7:0] fifo_out_data_2 = fifo_out_data_packed[23:16];
  wire signed [7:0] fifo_out_data_3 = fifo_out_data_packed[31:24];

  wire fifo_out_valid;
  wire fifo_out_ready;
  wire fifo_empty, fifo_full;
  fifo #(
      .DATA_WIDTH(DATA_IN_0_PRECISION_0 * WIDTH),  // 8*4 = 32
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
  // 2) 累计 4 行数据：使用 row_buffer 存储连续 4 行，每行 4 个数据
  // 将 row_buffer 定义为 signed
  logic signed [DATA_WIDTH-1:0] row_buffer[0:HEIGHT-1][0:WIDTH-1];
  logic [$clog2(HEIGHT+1)-1:0] row_count;

  // =========================================================
  // 3) 定义 4 个 2×2 池化窗口，每个窗口包含 4 个数据
  // window_regs 和 window_max 都声明为 signed
  logic signed [DATA_WIDTH-1:0] window_regs[0:NUM_WINDOWS-1][0:POOL_SIZE*POOL_SIZE-1];
  logic signed [DATA_WIDTH-1:0] window_max[0:NUM_WINDOWS-1];

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

  // =========================================================
  // 4) 状态机控制：IDLE, BUFFER, PROCESS, OUTPUT
  // =========================================================
  typedef enum logic [1:0] {
    IDLE,
    BUFFER,
    PROCESS,
    OUTPUT
  } state_t;
  state_t current_state, next_state;

  // 状态机控制信号
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
      // 初始化所有 row_buffer 行
      for (r = 0; r < HEIGHT; r = r + 1) for (c = 0; c < WIDTH; c = c + 1) row_buffer[r][c] <= 0;
    end else begin
      current_state    <= next_state;
      row_count        <= next_row_count;
      data_out_0_valid <= next_data_out_0_valid;
    end
  end

  // =========================================================
  // 5) 从 FIFO 读取数据，将一行数据存入 row_buffer
  // =========================================================
  assign fifo_out_ready = next_fifo_out_ready;
  always_ff @(posedge clk) begin
    if (!rst) begin
      if (fifo_out_ready && fifo_out_valid) begin
        for (int k = 0; k < WIDTH; k++) begin
          case (k)
            0: row_buffer[row_count-1][0] <= fifo_out_data_0;
            1: row_buffer[row_count-1][1] <= fifo_out_data_1;
            2: row_buffer[row_count-1][2] <= fifo_out_data_2;
            3: row_buffer[row_count-1][3] <= fifo_out_data_3;
          endcase
        end
      end
    end
  end

  // =========================================================
  // 6) PROCESS 状态：将累计的 4 行数据划分为 4 个 2×2 的 block
  //    Block A: row_buffer[0] 和 row_buffer[1] 的列 [0,1]
  //    Block B: row_buffer[0] 和 row_buffer[1] 的列 [2,3]
  //    Block C: row_buffer[2] 和 row_buffer[3] 的列 [0,1]
  //    Block D: row_buffer[2] 和 row_buffer[3] 的列 [2,3]
  // =========================================================
  always_ff @(posedge clk) begin
    if (!rst && current_state == PROCESS) begin
      // Block A
      window_regs[0][0] <= row_buffer[0][0];
      window_regs[0][1] <= row_buffer[0][1];
      window_regs[0][2] <= row_buffer[0][2];
      window_regs[0][3] <= row_buffer[0][3];
      // Block B
      window_regs[1][0] <= row_buffer[1][0];
      window_regs[1][1] <= row_buffer[1][1];
      window_regs[1][2] <= row_buffer[1][2];
      window_regs[1][3] <= row_buffer[1][3];
      // Block C
      window_regs[2][0] <= row_buffer[2][0];
      window_regs[2][1] <= row_buffer[2][1];
      window_regs[2][2] <= row_buffer[2][2];
      window_regs[2][3] <= row_buffer[2][3];
      // Block D
      window_regs[3][0] <= row_buffer[3][0];
      window_regs[3][1] <= row_buffer[3][1];
      window_regs[3][2] <= row_buffer[3][2];
      window_regs[3][3] <= row_buffer[3][3];
    end
  end

  always_ff @(posedge clk) begin
    if (!rst && current_state == OUTPUT) begin
      data_out_0[0] <= window_max[0];  // Block A 的最大值
      data_out_0[1] <= window_max[1];  // Block B 的最大值
      data_out_0[2] <= window_max[2];  // Block C 的最大值
      data_out_0[3] <= window_max[3];  // Block D 的最大值
    end
  end

endmodule
