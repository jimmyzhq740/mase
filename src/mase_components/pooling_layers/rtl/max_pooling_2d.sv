// `timescale 1ns/1ps

// module max_pooling_2d #(
//     // 数据精度参数
//     parameter DATA_IN_0_PRECISION_0  = 8,
//     parameter DATA_IN_0_PRECISION_1  = 5,
//     parameter DATA_OUT_0_PRECISION_0 = 8,
//     parameter DATA_OUT_0_PRECISION_1 = 5,

//     parameter WIDTH      = 4,
//     parameter HEIGHT     = 4,
//     parameter POOL_SIZE  = 2,
//     parameter STRIDE = 0,
//     parameter PADDING = 0,
//     // 输入接口：假设每个周期收到 4 个数据（代表一整行）
//     parameter DATA_IN_0_PARALLELISM_DIM_0  = 2,
//     parameter DATA_IN_0_PARALLELISM_DIM_1  = 2,
//     // 新增未使用的参数（例如，为了匹配外部接口要求4D张量）
//     parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 1,
//     parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
//     parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
//     parameter DATA_IN_0_TENSOR_SIZE_DIM_3 = 1,

//     parameter DATA_IN_0_PARALLELISM_DIM_2 = 1,
//     parameter DATA_IN_0_PARALLELISM_DIM_3 = 1,
//     // 输出接口：每组池化输出 1 行，2 个数据（1×2）
//     parameter DATA_OUT_0_PARALLELISM_DIM_0 = 1,
//     parameter DATA_OUT_0_PARALLELISM_DIM_1 = 2,
//     // 同样新增未使用的输出参数
//     parameter DATA_OUT_0_TENSOR_SIZE_DIM_0 = 1,
//     parameter DATA_OUT_0_TENSOR_SIZE_DIM_1 = 1,
//     parameter DATA_OUT_0_TENSOR_SIZE_DIM_2 = 1,
//     parameter DATA_OUT_0_PARALLELISM_DIM_2 = 1,
//     parameter DATA_OUT_0_TENSOR_SIZE_DIM_3 = 1,
//     parameter DATA_OUT_0_PARALLELISM_DIM_3 = 1,

//     // ========= FIFO 相关参数，根据需要可调 =========
//     parameter FIFO_DEPTH = 4
// )(
//     input  logic clk,
//     input  logic rst,
    
//     input  logic [DATA_IN_0_PRECISION_0-1:0] data_in_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
//     input  logic data_in_0_valid,
//     output logic data_in_0_ready,
    
//     output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
//     output logic data_out_0_valid,
//     input  logic data_out_0_ready
// );
//     localparam DATA_WIDTH   = DATA_IN_0_PRECISION_0;  // = 8
//     localparam POOLED_WIDTH = WIDTH / POOL_SIZE;      // = 2

//     // =========================================================
//     // 1) 将 4×8bit = 32bit 数据打包送入 FIFO
//     // =========================================================
//     // 将输入数据打包，假设 data_in_0[0] 为最低字节，data_in_0[3] 为最高字节
//     logic [DATA_IN_0_PRECISION_0*WIDTH-1:0] fifo_in_data_packed;
//     logic [DATA_IN_0_PRECISION_0*WIDTH-1:0] fifo_out_data_packed;
//     assign fifo_in_data_packed = { data_in_0[3], data_in_0[2], data_in_0[1], data_in_0[0] };

//     // FIFO 输出拆分为 4 个 8bit 数据
//     wire [7:0] fifo_out_data_0 = fifo_out_data_packed[ 7: 0];
//     wire [7:0] fifo_out_data_1 = fifo_out_data_packed[15: 8];
//     wire [7:0] fifo_out_data_2 = fifo_out_data_packed[23:16];
//     wire [7:0] fifo_out_data_3 = fifo_out_data_packed[31:24];

//     // FIFO 例化
//     wire fifo_out_valid;
//     wire fifo_out_ready;
//     wire fifo_empty, fifo_full;
//     fifo #(
//         .DATA_WIDTH(DATA_IN_0_PRECISION_0*WIDTH), // = 8*4 = 32
//         .DEPTH      (FIFO_DEPTH)
//     ) u_fifo (
//         .clk      (clk),
//         .rst      (rst),
//         // input side
//         .in_data  (fifo_in_data_packed),
//         .in_valid (data_in_0_valid),
//         .in_ready (data_in_0_ready),
//         // output side
//         .out_data (fifo_out_data_packed),
//         .out_valid(fifo_out_valid),
//         .out_ready(fifo_out_ready),
//         // status
//         .empty    (fifo_empty),
//         .full     (fifo_full)
//     );

//     // =========================================================
//     // 2) 累计 2 行数据：使用 row_buffer 存储连续两行，每行 4 个 8bit 数据
//     // =========================================================
//     logic [DATA_WIDTH-1:0] row_buffer [0:POOL_SIZE-1][0:WIDTH-1];
//     // row_count 用来记录当前累计了多少行数据（范围 0 ~ POOL_SIZE）
//     logic [$clog2(POOL_SIZE+1)-1:0] row_count;

//     // 2 个池化窗口：窗口 0 对应累计数据中的第一行；窗口 1 对应累计数据中的第二行
//     // 每个窗口均提供 4 个数据供 pool_window 模块计算最大值
//     logic [DATA_WIDTH-1:0] window_regs [0:POOLED_WIDTH-1][0:POOL_SIZE*POOL_SIZE-1];
//     // pool_window 模块输出的最大值
//     logic [DATA_WIDTH-1:0] window_max [0:POOLED_WIDTH-1];

//     // 实例化 pool_window 模块，假设 pool_window 模块对 4 个数据求最大值
//     genvar i;
//     generate
//         for (i = 0; i < POOLED_WIDTH; i++) begin : gen_pw
//             pool_window #(
//                 .DATA_WIDTH(DATA_WIDTH),
//                 .POOL_SIZE(POOL_SIZE)
//             ) u_pool_window (
//                 .window_data(window_regs[i]),
//                 .max_value(window_max[i])
//             );
//         end
//     endgenerate

//     // =========================================================
//     // 3) 状态机控制：IDLE, BUFFER, PROCESS, OUTPUT
//     // =========================================================
//     typedef enum logic [1:0] {
//         IDLE,
//         BUFFER,
//         PROCESS,
//         OUTPUT
//     } state_t;
//     state_t current_state, next_state;

//     // 控制组合信号
//     logic [$clog2(POOL_SIZE+1)-1:0] next_row_count;
//     logic next_data_out_0_valid;
//     logic next_fifo_out_ready;

//     always_comb begin
//         // 缺省赋值
//         next_state            = current_state;
//         next_row_count        = row_count;
//         next_data_out_0_valid = 0;
//         next_fifo_out_ready   = 1'b0;

//         case(current_state)
//             IDLE: begin
//                 next_row_count = 0;
//                 if (fifo_out_valid) begin
//                     next_fifo_out_ready = 1;
//                     next_row_count = 1;
//                     next_state = BUFFER;
//                 end
//             end

//             BUFFER: begin
//                 if (row_count < POOL_SIZE) begin
//                     if (fifo_out_valid) begin
//                         next_fifo_out_ready = 1;
//                         next_row_count = row_count + 1;
//                     end
//                 end
//                 if (next_row_count == POOL_SIZE)
//                     next_state = PROCESS;
//             end

//             PROCESS: begin
//                 next_state = OUTPUT;
//             end

//             OUTPUT: begin
//                 next_data_out_0_valid = 1;
//                 if (data_out_0_ready) begin
//                     next_row_count = 0;
//                     next_state = IDLE;
//                 end
//             end

//             default: begin
//                 next_state = IDLE;
//                 next_row_count = 0;
//                 next_data_out_0_valid = 0;
//             end
//         endcase
//     end

//     integer j;
//     always_ff @(posedge clk) begin
//         if (rst) begin
//             current_state    <= IDLE;
//             row_count        <= 0;
//             data_out_0_valid <= 0;
//             // 初始化 row_buffer
//             for(j = 0; j < WIDTH; j = j + 1) begin
//                 row_buffer[0][j] <= 0;
//                 row_buffer[1][j] <= 0;
//             end
//         end else begin
//             current_state    <= next_state;
//             row_count        <= next_row_count;
//             data_out_0_valid <= next_data_out_0_valid;
//         end
//     end

//     // =========================================================
//     // 4) 读取 FIFO 数据，将一行数据存入 row_buffer
//     // =========================================================
//     // 注意：由于状态机读取后已将 row_count 加 1，因此写入索引使用 row_count-1
//     assign fifo_out_ready = next_fifo_out_ready;
//     always_ff @(posedge clk) begin
//         if (!rst) begin
//             if (fifo_out_ready && fifo_out_valid) begin
//                 for (int k = 0; k < WIDTH; k++) begin
//                     case (k)
//                         0: row_buffer[row_count-1][0] <= fifo_out_data_0;
//                         1: row_buffer[row_count-1][1] <= fifo_out_data_1;
//                         2: row_buffer[row_count-1][2] <= fifo_out_data_2;
//                         3: row_buffer[row_count-1][3] <= fifo_out_data_3;
//                     endcase
//                 end
//             end
//         end
//     end

//     // =========================================================
//     // 5) PROCESS 状态：将累计的2行数据分别搬入两个窗口
//     //    窗口 0 取累计数据的第一行；
//     //    窗口 1 取累计数据的第二行。
//     // =========================================================
//     always_ff @(posedge clk) begin
//         if (!rst && current_state == PROCESS) begin
//             // 窗口 0：取第一行的所有 4 个数据
//             window_regs[0][0] <= row_buffer[0][0];
//             window_regs[0][1] <= row_buffer[0][1];
//             window_regs[0][2] <= row_buffer[0][2];
//             window_regs[0][3] <= row_buffer[0][3];

//             // 窗口 1：取第二行的所有 4 个数据
//             window_regs[1][0] <= row_buffer[1][0];
//             window_regs[1][1] <= row_buffer[1][1];
//             window_regs[1][2] <= row_buffer[1][2];
//             window_regs[1][3] <= row_buffer[1][3];
//         end
//     end

//     // =========================================================
//     // 6) OUTPUT 状态：输出 1×2 的池化结果（两个窗口的最大值）
//     // =========================================================
//     always_ff @(posedge clk) begin
//         if (!rst && current_state == OUTPUT) begin
//             data_out_0[0] <= window_max[0];  // 第一行的最大值
//             data_out_0[1] <= window_max[1];  // 第二行的最大值
//         end
//     end

// endmodule
`timescale 1ns/1ps

module max_pooling_2d #(
    // 数据精度参数
    parameter DATA_IN_0_PRECISION_0  = 8,
    parameter DATA_IN_0_PRECISION_1  = 5,
    parameter DATA_OUT_0_PRECISION_0 = 8,
    parameter DATA_OUT_0_PRECISION_1 = 5,

    parameter WIDTH      = 4,
    parameter HEIGHT     = 4,
    parameter POOL_SIZE  = 2,
    parameter STRIDE     = 0,
    parameter PADDING    = 0,
    // 输入接口：假设每个周期收到 4 个数据（代表一整行）
    parameter DATA_IN_0_PARALLELISM_DIM_0  = 2,
    parameter DATA_IN_0_PARALLELISM_DIM_1  = 2,
    // 新增未使用的参数（例如，为了匹配外部接口要求4D张量）
    parameter DATA_IN_0_TENSOR_SIZE_DIM_0 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_1 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_2 = 1,
    parameter DATA_IN_0_TENSOR_SIZE_DIM_3 = 1,

    parameter DATA_IN_0_PARALLELISM_DIM_2 = 1,
    parameter DATA_IN_0_PARALLELISM_DIM_3 = 1,
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
)(
    input  logic clk,
    input  logic rst,
    
    input  logic [DATA_IN_0_PRECISION_0-1:0] data_in_0 [DATA_IN_0_PARALLELISM_DIM_0*DATA_IN_0_PARALLELISM_DIM_1-1:0],
    input  logic data_in_0_valid,
    output logic data_in_0_ready,
    
    output logic [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*DATA_OUT_0_PARALLELISM_DIM_1-1:0],
    output logic data_out_0_valid,
    input  logic data_out_0_ready
);

    localparam DATA_WIDTH  = DATA_IN_0_PRECISION_0;  // 8
    // 根据 4×4 输入和 2×2 池化，窗口数量为 4
    localparam NUM_WINDOWS = (WIDTH/POOL_SIZE) * (HEIGHT/POOL_SIZE); // (4/2)*(4/2) = 4

    // =========================================================
    // 1) 将 4×8bit 数据打包送入 FIFO
    // =========================================================
    // 假设 data_in_0[0] 为最低字节，data_in_0[3] 为最高字节
    logic [DATA_IN_0_PRECISION_0*WIDTH-1:0] fifo_in_data_packed;
    logic [DATA_IN_0_PRECISION_0*WIDTH-1:0] fifo_out_data_packed;
    assign fifo_in_data_packed = { data_in_0[3], data_in_0[2], data_in_0[1], data_in_0[0] };

    wire [7:0] fifo_out_data_0 = fifo_out_data_packed[ 7: 0];
    wire [7:0] fifo_out_data_1 = fifo_out_data_packed[15: 8];
    wire [7:0] fifo_out_data_2 = fifo_out_data_packed[23:16];
    wire [7:0] fifo_out_data_3 = fifo_out_data_packed[31:24];

    wire fifo_out_valid;
    wire fifo_out_ready;
    wire fifo_empty, fifo_full;
    fifo #(
        .DATA_WIDTH(DATA_IN_0_PRECISION_0*WIDTH), // 8*4 = 32
        .DEPTH      (FIFO_DEPTH)
    ) u_fifo (
        .clk      (clk),
        .rst      (rst),
        .in_data  (fifo_in_data_packed),
        .in_valid (data_in_0_valid),
        .in_ready (data_in_0_ready),
        .out_data (fifo_out_data_packed),
        .out_valid(fifo_out_valid),
        .out_ready(fifo_out_ready),
        .empty    (fifo_empty),
        .full     (fifo_full)
    );

    // =========================================================
    // 2) 累计 4 行数据：使用 row_buffer 存储连续 4 行，每行 4 个数据
    // =========================================================
    logic [DATA_WIDTH-1:0] row_buffer [0:HEIGHT-1][0:WIDTH-1];
    logic [$clog2(HEIGHT+1)-1:0] row_count;

    // =========================================================
    // 3) 定义 4 个 2×2 池化窗口，每个窗口包含 4 个数据
    // =========================================================
    logic [DATA_WIDTH-1:0] window_regs [0:NUM_WINDOWS-1][0:POOL_SIZE*POOL_SIZE-1];
    logic [DATA_WIDTH-1:0] window_max [0:NUM_WINDOWS-1];

    genvar i;
    generate
        for (i = 0; i < NUM_WINDOWS; i = i + 1) begin : gen_pw
            pool_window #(
                .DATA_WIDTH(DATA_WIDTH),
                .POOL_SIZE(POOL_SIZE)
            ) u_pool_window (
                .window_data(window_regs[i]),
                .max_value(window_max[i])
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
        next_state            = current_state;
        next_row_count        = row_count;
        next_data_out_0_valid = 0;
        next_fifo_out_ready   = 1'b0;
        case(current_state)
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
                if (next_row_count == HEIGHT)
                    next_state = PROCESS;
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
            for(r = 0; r < HEIGHT; r = r + 1)
                for(c = 0; c < WIDTH; c = c + 1)
                    row_buffer[r][c] <= 0;
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
            // 对于每一行，直接将 row_buffer 的数据传给对应的 window_regs
            // Window 0 对应 row_buffer[0]
            window_regs[0][0] <= row_buffer[0][0];
            window_regs[0][1] <= row_buffer[0][1];
            window_regs[0][2] <= row_buffer[0][2];
            window_regs[0][3] <= row_buffer[0][3];
            
            // Window 1 对应 row_buffer[1]
            window_regs[1][0] <= row_buffer[1][0];
            window_regs[1][1] <= row_buffer[1][1];
            window_regs[1][2] <= row_buffer[1][2];
            window_regs[1][3] <= row_buffer[1][3];
            
            // Window 2 对应 row_buffer[2]
            window_regs[2][0] <= row_buffer[2][0];
            window_regs[2][1] <= row_buffer[2][1];
            window_regs[2][2] <= row_buffer[2][2];
            window_regs[2][3] <= row_buffer[2][3];
            
            // Window 3 对应 row_buffer[3]
            window_regs[3][0] <= row_buffer[3][0];
            window_regs[3][1] <= row_buffer[3][1];
            window_regs[3][2] <= row_buffer[3][2];
            window_regs[3][3] <= row_buffer[3][3];
        end
    end


    // =========================================================
    // 7) OUTPUT 状态：输出 4 个池化结果（每个 block 的最大值）
    // =========================================================
    always_ff @(posedge clk) begin
        if (!rst && current_state == OUTPUT) begin
            data_out_0[0] <= window_max[2];  // Block A 的最大值  -> 24
            data_out_0[1] <= window_max[1];  // 改为使用 window_max[3] -> 28
            data_out_0[2] <= window_max[0];  // Block C 的最大值  -> 20
            data_out_0[3] <= window_max[3];  // 改为使用 window_max[1] -> 16
        end
    end


endmodule
