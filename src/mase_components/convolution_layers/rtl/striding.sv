module striding #(
    // 可变参数定义
    parameter DATA_WIDTH  = 8,
    parameter BATCH_SIZE  = 1,  // 此处 testbench 用单个 batch
    parameter MATRIX_ROWS = 8,  // padding前 row 数
    parameter MATRIX_COLS = 8,  // padding前 column 数
    parameter KERNEL_ROWS = 3,  // sliding window 行数
    parameter KERNEL_COLS = 3,  // sliding window 列数
    parameter STRIDE      = 1   // 步长（本例未用到，因为输出顺序由分组控制）
) (
    input logic clk,
    input logic rst,


    input  logic                  in_valid,
    output logic                  in_ready,
    input  logic [DATA_WIDTH-1:0] data_in,


    output logic                  out_valid,
    input  logic                  out_ready,
    output logic [DATA_WIDTH-1:0] data_out [0:(KERNEL_ROWS*KERNEL_COLS*BATCH_SIZE)-1]
);


  localparam P_ROWS = MATRIX_ROWS + 2;  // padding 后行数
  localparam P_COLS = MATRIX_COLS + 2;  // padding 后列数
  localparam TOTAL_PIXELS = P_ROWS * P_COLS;  // 输入总像素数

  // 有效窗口的顶行/左列个数
  localparam VALID_ROWS = P_ROWS - KERNEL_ROWS + 1;
  localparam VALID_COLS = P_COLS - KERNEL_COLS + 1;


  // 使用 ceil 除法：例如 VALID_ROWS=8时，ROW_GROUPS=4；若为奇数则最后一组只有1行
  localparam integer ROW_GROUPS = (VALID_ROWS + 1) / 2;
  localparam integer COL_GROUPS = (VALID_COLS + 1) / 2;
  // 总窗口数 = VALID_ROWS * VALID_COLS
  localparam integer TOTAL_WINDOWS = VALID_ROWS * VALID_COLS;

  // FSM 状态定义 - 修改为允许同时接收和输出

  typedef enum logic [1:0] {
    CAPTURE   = 2'd0,  // 仅采集输入数据
    PROC_OUT  = 2'd1,  // 边接收边输出
    FLUSH_OUT = 2'd2,  // 仅输出剩余窗口
    DONE      = 2'd3   // 所有窗口数据输出完毕，等待新输入
  } state_t;
  state_t state, next_state;


  logic [DATA_WIDTH-1:0] frame_mem[0:P_ROWS-1][0:P_COLS-1];

  // 输入计数器
  integer in_count;

  // 接收到的完整行数跟踪
  integer received_rows;


  integer rg, cg, sw, win_count;


  // 输出端 data_out 直接由 window_reg 驱动

  logic [DATA_WIDTH-1:0] window_reg[0:(KERNEL_ROWS*KERNEL_COLS*BATCH_SIZE)-1];
  genvar i;
  generate
    for (i = 0; i < KERNEL_ROWS * KERNEL_COLS * BATCH_SIZE; i = i + 1) begin : assign_data_out
      assign data_out[i] = window_reg[i];
    end
  endgenerate

  // in_ready：在接收状态下有效
  assign in_ready = (state == CAPTURE) || (state == PROC_OUT) || (state == DONE);


  // 状态转移组合逻辑

  always_comb begin
    case (state)
      CAPTURE: begin
        // 当接收到足够行数以开始输出第一个窗口时转为PROC_OUT状态
        if (received_rows >= KERNEL_ROWS) next_state = PROC_OUT;
        else next_state = CAPTURE;
      end
      PROC_OUT: begin
        // 当所有输入接收完毕，但还有窗口需要输出时转为FLUSH_OUT
        if (in_count == TOTAL_PIXELS) next_state = FLUSH_OUT;
        else next_state = PROC_OUT;
      end
      FLUSH_OUT: begin
        // 当所有窗口输出完毕后转到DONE状态
        if (win_count == TOTAL_WINDOWS) next_state = DONE;
        else next_state = FLUSH_OUT;
      end
      DONE: begin
        // DONE状态下若检测到新输入，则重新采集新帧
        if (in_valid) next_state = CAPTURE;
        else next_state = DONE;
      end
      default: next_state = CAPTURE;
    endcase
  end


  // 状态机和计数器更新（时钟沿更新）

  always_ff @(posedge clk) begin
    if (rst) begin
      state         <= CAPTURE;
      in_count      <= 0;
      received_rows <= 0;
      win_count     <= 0;
      rg            <= 0;
      cg            <= 0;
      sw            <= 0;
      out_valid     <= 0;

      // 初始化frame_mem（输入缓冲区）为0
      for (int r = 0; r < P_ROWS; r++) begin
        for (int c = 0; c < P_COLS; c++) begin
          frame_mem[r][c] <= '0;
        end
      end
    end else begin
      state <= next_state;

      // 输入处理逻辑
      if ((state == CAPTURE || state == PROC_OUT || state == DONE) && in_valid && in_ready) begin
        // 计算当前写入的行和列
        automatic int curr_row = in_count / P_COLS;
        automatic int curr_col = in_count % P_COLS;

        // 写入数据
        frame_mem[curr_row][curr_col] <= data_in;
        in_count <= in_count + 1;

        // 当完成一整行时更新received_rows
        if (curr_col == P_COLS - 1) received_rows <= received_rows + 1;

        // 重置状态
        if (state == DONE) begin
          win_count <= 0;
          rg <= 0;
          cg <= 0;
          sw <= 0;
          received_rows <= (curr_col == P_COLS-1) ? 1 : 0; // 如果刚好完成一行则为1，否则为0
        end
      end

      // 窗口输出处理逻辑
      if (state == PROC_OUT || state == FLUSH_OUT) begin
        // 当前行组的基准行索引
        automatic int base_row = rg * 2;

        // 只有当我们接收了足够的行才处理输出
        if (received_rows >= base_row + KERNEL_ROWS || state == FLUSH_OUT) begin
          // 当窗口数据已通过握手输出，更新窗口计数
          if (out_valid && out_ready) begin
            out_valid <= 0;
            win_count <= win_count + 1;

            // 更新输出索引
            if (sw < 3) begin
              sw <= sw + 1;
            end else begin
              sw <= 0;
              if (cg < COL_GROUPS - 1) cg <= cg + 1;
              else begin
                cg <= 0;
                if (rg < ROW_GROUPS - 1) rg <= rg + 1;
                else;  // 已到最后一组，win_count判断会使状态转DONE
              end
            end
          end
        end
      end

      // DONE状态处理
      if (state == DONE) begin
        if (in_valid) begin
          in_count <= 0;
        end
        out_valid <= 0;
      end
    end
  end


  // sliding window 数据提取及 out_valid 控制

  always_ff @(posedge clk) begin
    if (rst) begin
      out_valid <= 0;
      // 初始化 window_reg
      for (int j = 0; j < KERNEL_ROWS * KERNEL_COLS * BATCH_SIZE; j = j + 1) window_reg[j] <= '0;
    end else begin
      // 在PROC_OUT或FLUSH_OUT状态下处理窗口输出
      if (state == PROC_OUT || state == FLUSH_OUT) begin
        // 当前行组的基准行
        automatic int base_row = rg * 2;

        // 只有当我们有足够的行数据时才输出窗口
        if (received_rows >= base_row + KERNEL_ROWS || state == FLUSH_OUT) begin
          // 若未拉高 out_valid，则加载当前窗口数据
          if (!out_valid) begin
            int curr_row, curr_col;

            // 根据当前组及子索引 sw 计算窗口顶行与左列
            curr_row = rg * 2 + ((sw < 2) ? 0 : 1);
            curr_col = cg * 2 + ((sw % 2 == 0) ? 0 : 1);

            // 如果计算得到的位置超出范围，则输出窗口全0
            // 否则从frame_mem中提取窗口数据
            if ((curr_row >= VALID_ROWS) || (curr_col >= VALID_COLS)) begin
              for (int idx = 0; idx < KERNEL_ROWS * KERNEL_COLS * BATCH_SIZE; idx = idx + 1)
                window_reg[idx] <= '0;
            end else begin
              int idx;
              idx = 0;
              for (int r = 0; r < KERNEL_ROWS; r = r + 1) begin
                for (int c = 0; c < KERNEL_COLS; c = c + 1) begin
                  window_reg[idx] <= frame_mem[curr_row+r][curr_col+c];
                  idx = idx + 1;
                end
              end
            end
            out_valid <= 1;
          end
        end
      end else begin
        out_valid <= 0;
      end
    end
  end

endmodule
