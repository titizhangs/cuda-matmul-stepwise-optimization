#include "list_matmul.cuh"

// ============================================================================
// Tiled: 共享内存分块 + K向跨步加载
// ============================================================================
#define BLOCK_M 8
#define BLOCK_N 32
#define BLOCK_K 64

static __global__ void matmul_basic_tiled(const float *__restrict__ A,
                                          const float *__restrict__ B,
                                          float *__restrict__ C, int N) {

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int g_row = blockIdx.y * BLOCK_M + ty;
  const int g_col = blockIdx.x * BLOCK_N + tx;

  __shared__ float sA[BLOCK_M][BLOCK_K];
  __shared__ float sB[BLOCK_K][BLOCK_N];

  float accum = 0.0f;

  // 遍历K方向所有分块
  const int num_tiles_k = (N + BLOCK_K - 1) / BLOCK_K;
  for (int tile_k = 0; tile_k < num_tiles_k; ++tile_k) {
    // 当前K分块的全局基址偏移
    const size_t k_base = (size_t)tile_k * BLOCK_K;

    // 加载A分块到共享内存，沿K方向跨步填充
    // 沿行K方向以BLOCK_N为步长循环填充
    const size_t a_row = g_row;
    for (int k = tx; k < BLOCK_K; k += BLOCK_N) {
      const size_t a_col = k_base + k;
      if (a_row < N && a_col < N) {
        const size_t a_idx = (size_t)a_row * N + a_col;
        sA[ty][k] = A[a_idx];
      } else {
        sA[ty][k] = 0.0f;
      }
    }
    // 加载B分块到共享内存，沿K方向跨步填充
    // 沿行K方向以BLOCK_M为步长循环填充
    const size_t b_col = g_col;
    for (int k = ty; k < BLOCK_K; k += BLOCK_M) {
      const size_t b_row = k_base + k;
      if (b_row < N && b_col < N) {
        const size_t b_idx = (size_t)b_row * N + b_col;
        sB[k][tx] = B[b_idx];
      } else {
        sB[k][tx] = 0.0f;
      }
    }

    // 等待所有线程加载完成
    __syncthreads();

// Tile内矩阵乘，内积累加
#pragma unroll
    for (int k = 0; k < BLOCK_K; ++k) {
      accum += sA[ty][k] * sB[k][tx];
    }

    // 等待所有线程计算完成，再下一轮覆盖共享内存
    __syncthreads();
  }

  if (g_row < N && g_col < N) {
    const size_t c_idx = (size_t)g_row * N + g_col;
    C[c_idx] = accum;
  }
}

void v3_matmul_basic_tiled_K64(const float *d_A, const float *d_B, float *d_C,
                               int N) {
  dim3 block(BLOCK_N, BLOCK_M);
  dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (N + BLOCK_M - 1) / BLOCK_M);

  matmul_basic_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
}
