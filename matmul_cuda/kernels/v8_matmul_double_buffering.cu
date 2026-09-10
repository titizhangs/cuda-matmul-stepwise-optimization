#include "list_matmul.cuh"

// ============================================================================
// Tiled: 共享内存分块 + K向跨步加载kTileK + 寄存器分块kThreadM,kThreadN +
// 双缓冲（粗粒度）
// ============================================================================
template <int kBlockN, int kBlockM, int kThreadM, int kThreadN, int kTileK,
          int kTileM, int kTileN>
static __global__ void MatMulDoubleBuffering(const float *__restrict__ A,
                                             const float *__restrict__ B,
                                             float *__restrict__ C, int N) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int gRowBase = blockIdx.y * kTileM + ty * kThreadM;
  const int gColBase = blockIdx.x * kTileN + tx * kThreadN;

  __shared__ float sA0[kTileM][kTileK];
  __shared__ float sB0[kTileK][kTileN];

  __shared__ float sA1[kTileM][kTileK];
  __shared__ float sB1[kTileK][kTileN];

  float acc[kThreadM][kThreadN] = {0.0f};

  int cur = 0;
  int next = 1;

  // 预加载tile 0到sA0/sB0
  //  加载A分块到共享内存，沿K方向跨步填充
  //  沿行K方向以BLOCK_N为步长循环填充
  for (int k = tx; k < kTileK; k += kBlockN) {
    const size_t aCol = 0 + k;
    for (int m = 0; m < kThreadM; m++) {
      const size_t aRow = gRowBase + m;
      if (aRow < N && aCol < N) {
        const size_t aIdx = (size_t)aRow * N + aCol;
        sA0[kThreadM * ty + m][k] = A[aIdx];
      } else {
        sA0[kThreadM * ty + m][k] = 0.0f;
      }
    }
  }

  // 加载B分块到共享内存，沿K方向跨步填充
  // 沿行K方向以BLOCK_M为步长循环填充
  for (int k = ty; k < kTileK; k += kBlockM) {
    const size_t bRow = 0 + k;
    for (int n = 0; n < kThreadN; n++) {
      const size_t bCol = gColBase + n;
      if (bRow < N && bCol < N) {
        const size_t bIdx = (size_t)bRow * N + bCol;
        sB0[k][kThreadN * tx + n] = B[bIdx];
      } else {
        sB0[k][kThreadN * tx + n] = 0.0f;
      }
    }
  }
  __syncthreads();

  // 遍历K方向所有分块
  const int numTilesK = (N + kTileK - 1) / kTileK;
  for (int tileK = 0; tileK < numTilesK; ++tileK) {
    if (tileK + 1 < numTilesK) {
      // 当前K分块的全局基址偏移
      const size_t tileKBase = (size_t)(tileK + 1) * kTileK;
      if (next == 0) {
        // 加载A分块到共享内存，沿K方向跨步填充
        // 沿行K方向以BLOCK_N为步长循环填充
        for (int k = tx; k < kTileK; k += kBlockN) {
          const size_t aCol = tileKBase + k;
          for (int m = 0; m < kThreadM; m++) {
            const size_t aRow = gRowBase + m;
            if (aRow < N && aCol < N) {
              const size_t aIdx = (size_t)aRow * N + aCol;
              sA0[kThreadM * ty + m][k] = A[aIdx];
            } else {
              sA0[kThreadM * ty + m][k] = 0.0f;
            }
          }
        }

        // 加载B分块到共享内存，沿K方向跨步填充
        // 沿行K方向以BLOCK_M为步长循环填充
        for (int k = ty; k < kTileK; k += kBlockM) {
          const size_t bRow = tileKBase + k;
          for (int n = 0; n < kThreadN; n++) {
            const size_t bCol = gColBase + n;
            if (bRow < N && bCol < N) {
              const size_t bIdx = (size_t)bRow * N + bCol;
              sB0[k][kThreadN * tx + n] = B[bIdx];
            } else {
              sB0[k][kThreadN * tx + n] = 0.0f;
            }
          }
        }

      } else {
        // 加载A分块到共享内存，沿K方向跨步填充
        // 沿行K方向以BLOCK_N为步长循环填充
        for (int k = tx; k < kTileK; k += kBlockN) {
          const size_t aCol = tileKBase + k;
          for (int m = 0; m < kThreadM; m++) {
            const size_t aRow = gRowBase + m;
            if (aRow < N && aCol < N) {
              const size_t aIdx = (size_t)aRow * N + aCol;
              sA1[kThreadM * ty + m][k] = A[aIdx];
            } else {
              sA1[kThreadM * ty + m][k] = 0.0f;
            }
          }
        }

        // 加载B分块到共享内存，沿K方向跨步填充
        // 沿行K方向以BLOCK_M为步长循环填充
        for (int k = ty; k < kTileK; k += kBlockM) {
          const size_t bRow = tileKBase + k;
          for (int n = 0; n < kThreadN; n++) {
            const size_t bCol = gColBase + n;
            if (bRow < N && bCol < N) {
              const size_t bIdx = (size_t)bRow * N + bCol;
              sB1[k][kThreadN * tx + n] = B[bIdx];
            } else {
              sB1[k][kThreadN * tx + n] = 0.0f;
            }
          }
        }
      }
    }

    // Tile内矩阵乘，内积累加+寄存器分块
    // A数组沿M方向
    float aM[kThreadM];
    // B数组沿N方向
    float bN[kThreadN];
    if (cur == 0) {
#pragma unroll
      for (int k = 0; k < kTileK; k++) {
#pragma unroll
        for (int m = 0; m < kThreadM; m++) {
          aM[m] = sA0[kThreadM * ty + m][k];
        }
#pragma unroll
        for (int n = 0; n < kThreadN; n++) {
          bN[n] = sB0[k][kThreadN * tx + n];
        }
#pragma unroll
        for (int m = 0; m < kThreadM; m++) {
          for (int n = 0; n < kThreadN; n++) {
            acc[m][n] += aM[m] * bN[n];
          }
        }
      }
    } else {
#pragma unroll
      for (int k = 0; k < kTileK; k++) {
#pragma unroll
        for (int m = 0; m < kThreadM; m++) {
          aM[m] = sA1[kThreadM * ty + m][k];
        }
#pragma unroll
        for (int n = 0; n < kThreadN; n++) {
          bN[n] = sB1[k][kThreadN * tx + n];
        }
#pragma unroll
        for (int m = 0; m < kThreadM; m++) {
          for (int n = 0; n < kThreadN; n++) {
            acc[m][n] += aM[m] * bN[n];
          }
        }
      }
    }
    // 等待所有线程计算完成，再下一轮覆盖共享内存
    __syncthreads();
    cur ^= 1;
    next ^= 1;
  }
  for (int m = 0; m < kThreadM; m++) {
    int cRow = gRowBase + m;
    for (int n = 0; n < kThreadN; n++) {
      int cCol = gColBase + n;
      if (cRow < N && cCol < N) {
        const size_t cIdx = (size_t)cRow * N + cCol;
        C[cIdx] = acc[m][n];
      }
    }
  }
}

void V8MatMulDoubleBuffering(const float *dA, const float *dB, float *dC,
                             int N) {

  constexpr int kBlockN = 32;
  constexpr int kBlockM = 8;
  constexpr int kThreadM = 4;
  constexpr int kThreadN = 4;
  constexpr int kTileK = 32;

  constexpr int kTileM = kBlockM * kThreadM;
  constexpr int kTileN = kBlockN * kThreadN;

  dim3 block(kBlockN, kBlockM);
  dim3 grid((N + kTileN - 1) / kTileN, (N + kTileM - 1) / kTileM);

  MatMulDoubleBuffering<kBlockN, kBlockM, kThreadM, kThreadN, kTileK, kTileM,
                        kTileN><<<grid, block>>>(dA, dB, dC, N);
}
