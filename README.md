
# CUDA SGEMM 逐阶优化：RTX 5070 Ti 性能达 cuBLAS 79%

**技术覆盖**：完整覆盖内存分层优化与全尺寸性能权衡 
**工具链**：CUDA Toolkit、Nsight Compute、Compute Sanitizer、CMake
# 项目概述
本项目为矩阵优化项目（使用RTX5070ti消费级显卡），使用ncu分析优化方向，compute-sanitizer定位错误，从朴素矩阵逐级优化最终在达到了cublasSgemm性能的79%左右(4096*4096*4096无边界方阵乘法)
# 核心优化点
**全局内存层**
- 行向 float4 向量化加载，对齐 128 字节缓存行，实现 Warp 级完美合并访问，打满内存带宽；
- 共享内存分块计算驱动分块加载，将全局内存访问总量从 O (n³) 降低至 O (n³ / TileSize)，计算访存比提升 TileSize 倍，大幅掩盖全局内存访问延迟。
**共享内存层**
- 共享内存分块（Tiling），将（B数组）全局内存的列访问转换为共享内存的行访问；
- 控制访问步长降低 Bank 冲突，经验证为消冲突会引入额外写指令开销，整体负收益，最终保留 4 路冲突的均衡方案；
- 验证了共享内存双缓冲（乒乓预加载）方案，当前 Tile 尺寸下共享内存翻倍，占用率下降，综合收益不明显故未采用。
**寄存器计算层**
- 线程粗化 + 寄存器分块，每个线程维护多元素累加器，提升计算密度，减少共享内存访问次数；
# 性能数据
测试条件：RTX 5070 Ti，纯 FP32 计算，cudaEvent 硬件计时，预热后多次运行取平均值，对齐尺寸无边界处理

| 矩阵尺寸 N | cuBLAS 耗时 | 本项目v13版本耗时 | 性能占比  |
| ------ | --------- | ---------- | ----- |
| 512    | 0.029 ms  | 0.059 ms   | 49.2% |
| 1024   | 0.092 ms  | 0.115 ms   | 80%   |
| 2048   | 0.569 ms  | 0.710 ms   | 80.1% |
| 4096   | 4.340 ms  | 5.477 ms   | 79.2% |
| 8192   | 33.938 ms | 43.693 ms  | 77.7% |

备注：当前版本固定了分块 Tile 尺寸、单 Block 线程数与线程粗化系数，核心面向大尺寸矩阵做峰值性能优化。小尺寸场景下总线程块数不足，硬件利用率偏低：512 尺寸下总 Block 数仅 16 个，远低于 RTX 5070 Ti 的 70 个 SM 硬件并行规模，多数 SM 全程闲置；1024 尺寸下总 Block 数为 64 个，仍有少量 SM 全程空闲未被利用。
若针对小尺寸专项优化，可通过缩小 Tile 尺寸增加总 Block 数，同时匹配调整单 Block 线程数与粗化系数，让 SM 满载以提升性能；但该优化需要额外增加尺寸适配分支，且对本项目核心目标增益有限，因此当前版本以大尺寸矩阵峰值性能为优先，未针对小尺寸做定向适配。
# 最优版本核心代码片段
共享内存加载阶段  float4读写，线程任务划分坐标重定义(block单维)，同一warp完美合并访问，共享内存sA，sB存在4way冲突，经验证若为消冲突做转置会引入额外写指令与调度开销，整体性能负收益，因此保留当前方案，属于复杂度与性能的合理权衡。
```cuda
  for (int tileK = 0; tileK < (N + kTileK - 1) / kTileK; tileK++) {
    // 加载128*32
    // 计算block内坐标，2^8线程，(优先满足行方向float4)其中2^3用来作为k方向坐标，剩下的2^5用来作为M方向坐标
    // 128/(2^5)= 4轮循环加载完tile
    int coordCol = threadIdx.x % (kTileK / 4);
    int coordRow = threadIdx.x / (kTileK / 4);
    constexpr int kCoordMMax = kThreads / (kTileK / 4);
    for (int m = 0; m < (kBlockOutMN + kCoordMMax - 1) / kCoordMMax; m++) {
      float4 aVec = *reinterpret_cast<const float4 *>(
          &A[(blockIdx.y * kBlockOutMN + m * kCoordMMax + coordRow) * N +
             (tileK * kTileK + coordCol * 4)]);
      reinterpret_cast<float4 *>(
          &sA0[m * kCoordMMax + coordRow][coordCol * 4])[0] = aVec;
    }
    // 加载32 * 128
    // 计算block内坐标，2^8线程，(优先满足行方向float4)其中2^5用来作为N方向坐标，剩下的2^3用来作为k方向坐标
    // 32/(2^3)= 4轮循环加载完tile
    coordCol = threadIdx.x % (kBlockOutMN / 4);
    coordRow = threadIdx.x / (kBlockOutMN / 4);
    constexpr int kCoordKMax = kThreads / (kBlockOutMN / 4);
    for (int k = 0; k < (kTileK + kCoordKMax - 1) / kCoordKMax; k++) {
      float4 bVec = *reinterpret_cast<const float4 *>(
          &B[(tileK * kTileK + k * kCoordKMax + coordRow) * N +
             (blockIdx.x * kBlockOutMN + coordCol * 4)]);
      reinterpret_cast<float4 *>(
          &sB0[k * kCoordKMax + coordRow][coordCol * 4])[0] = bVec;
    }
```
计算阶段 线程粗化+寄存器分块，编译循环展开float4向量合并（sB0的行向连续sB0自动合并，外层通过编译指示强制展开 4 次并开启向量化，使sA跨k行向连续合并为float4）
```cuda
#pragma unroll 4
#pragma loop vectorize(enable)
    for (int k = 0; k < kTileK; k++) {
      for (int m = 0; m < kThreadCoarsening; m++) {
        //2路冲突+广播，可接受
        float tmpA = sA0[row * kThreadCoarsening + m][k];
        for (int n = 0; n < kThreadCoarsening; n++) {
          float tmpB = sB0[k][col * kThreadCoarsening + n];
          acc[m][n] += tmpA * tmpB;
        }
      }
    }
```

# 运行方法
可执行文件：`matmul_bench.exe`
- `--kernel`：选择内核版本，取值 0-13，v13 为最终最优版本；99 为 cuBLAS 对照版本
- `-n`：设置方阵边长
- `--allCheck yes`：开启所有结果元素正确性校验，打印预期值与实际值，配合小尺寸用于错误定位
- `--warmRound`：预热次数
- `--runRound`：预热后的执行次数

# 项目结构
项目按 v1 到 v13 逐版本迭代优化，各个版本ncu报告在ncu-report下面，最终以v13为最终优化版本(v13版本自身有多次迭代，见v13的ncu报告)
```
.
└── matmul_cuda
    ├── CMakeLists.txt
    ├── benchmark.cu                     # 性能测试与校验入口
    ├── kernels                          # 各版本内核实现
    │   ├── cublas_gemm.cu               # cuBLAS 基准对照
    │   ├── list_matmul.cuh
    │   ├── naive_matmul.cu              #朴素矩阵乘法
    │   ├── v1_naive_matmul.cu           #调整block后朴素矩阵
    │   ├── v2_matmul_basic_tiled.cu     #共享内存分块
    │   ├── ...
    │   ├── v5_matmul_register_tiled.cu  #寄存器分块
    │   ├── ...
    │   ├── v8_matmul_double_buffering.cu #引入双缓冲区
    │   ├── v9_matmul_float4.cu          #float4使用
    │   ├── v10_matmul_wmma.cu           #wmma版本
    │   ├── ...
    │   └── v13_matmul.cu   #综合使用终版，过程版本见v13的24个ncu报告(里面有代码)
    ├── naive_matmul.exe
    ├── naive_matmul.ncu-rep             #朴素矩阵乘法报告
    ├── naive_matmul.nsys-rep
    └── ncu-report                       # 各版本对应的ncu报告
```
