#pragma once

// ============= 总开关 =============
// 注释掉这行，所有调试代码块编译时直接消失，零性能开销
#define CUDA_ENABLE_DEBUG

#ifdef CUDA_ENABLE_DEBUG
    // 无条件执行调试块：后面直接跟 { ... }
    #define CUDA_DEBUG_BLOCK \
        for (int _cuda_debug_ = 1; _cuda_debug_; _cuda_debug_ = 0)

    // 条件执行调试块：只有满足 cond 的线程才进入代码块
    #define CUDA_DEBUG_BLOCK_IF(cond) \
        for (int _cuda_debug_ = (cond); _cuda_debug_; _cuda_debug_ = 0)
#else
    // 关闭时展开为死循环体，编译器直接整段优化删除
    #define CUDA_DEBUG_BLOCK \
        for (int _cuda_debug_ = 0; _cuda_debug_; _cuda_debug_ = 0)
    #define CUDA_DEBUG_BLOCK_IF(cond) \
        for (int _cuda_debug_ = 0; _cuda_debug_; _cuda_debug_ = 0)
#endif


void CublasGemm(const float* d_A, const float* d_B, float* d_C, int N);

void cuda_naive_matmul(const float* d_A, const float* d_B, float* d_C, int N);
void v1_naive_matmul(const float* d_A, const float* d_B, float* d_C, int N);
void v2_matmul_basic_tiled(const float* d_A, const float* d_B, float* d_C, int N);
void v3_matmul_basic_tiled_K64(const float* d_A, const float* d_B, float* d_C, int N);
void v4_matmul_basic_tiled_K128(const float* d_A, const float* d_B, float* d_C, int N);
void V5MatMulRegisterTiled(const float* d_A, const float* d_B, float* d_C, int N);
void V6MatMulRegisterTiled(const float* d_A, const float* d_B, float* d_C, int N);
void V7MatMulRegisterTiled(const float* d_A, const float* d_B, float* d_C, int N);
void V8MatMulDoubleBuffering(const float* d_A, const float* d_B, float* d_C, int N);
void V9MatMulFloat4(const float *dA, const float *dB, float *dC, int N);
void V10MatMulWMMA(const float *dA, const float *dB, float *dC, int N);
void V11MatMulWMMAOpt(const float *dA, const float *dB, float *dC, int N);
void V12MatMul(const float *dA, const float *dB, float *dC, int N);
void V13MatMul(const float *dA, const float *dB, float *dC, int N);


