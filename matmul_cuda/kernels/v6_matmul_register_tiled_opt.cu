#include "list_matmul.cuh"

// ============================================================================
// Tiled: 共享内存分块 + K向跨步加载kTileK + 寄存器分块kThreadM,kThreadN
//  加载B分块到共享内存优化，合并内存访问+同时修复了sB写入的bankconflict
// ============================================================================
template <int kBlockN, int kBlockM, int kThreadM, int kThreadN, int kTileK,
          int kTileM, int kTileN>
static __global__ void MatMulRegisterTiled(const float *__restrict__ A,
                                           const float *__restrict__ B,
                                           float *__restrict__ C, int N) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int gRowBase = blockIdx.y * kTileM + ty * kThreadM;
  const int gColBase = blockIdx.x * kTileN + tx * kThreadN;

  __shared__ float sA[kTileM][kTileK];
  __shared__ float sB[kTileK][kTileN];

  float acc[kThreadM][kThreadN] = {0.0f};

  // 遍历K方向所有分块
  const int numTilesK = (N + kTileK - 1) / kTileK;
  for (int tileK = 0; tileK < numTilesK; ++tileK) {
    // 当前K分块的全局基址偏移
    const size_t tileKBase = (size_t)tileK * kTileK;

    // 加载A分块到共享内存，沿K方向跨步填充
    // 沿行K方向以BLOCK_N为步长循环填充
    for (int k = tx; k < kTileK; k += kBlockN) {
      const size_t aCol = tileKBase + k;
      // 单线程处理连续的元素加载
      for (int m = 0; m < kThreadM; m++) {
        const size_t aRow = gRowBase + m;
        if (aRow < N && aCol < N) {
          const size_t aIdx = (size_t)aRow * N + aCol;
          sA[kThreadM * ty + m][k] = A[aIdx];
        } else {
          sA[kThreadM * ty + m][k] = 0.0f;
        }
      }
    }

    // 加载B分块到共享内存，沿K方向跨步填充
    // 沿行K方向以BLOCK_M为步长循环填充
    for (int k = ty; k < kTileK; k += kBlockM) {
      const size_t bRow = tileKBase + k;
      // 单线程处理跨步的元素加载，以使warp内线程间访问连续内存合并。
      for (int n = tx; n < kTileN; n+=kBlockN) {
        const size_t bCol = blockIdx.x * kTileN + n;
        if (bRow < N && bCol < N) {
          const size_t bIdx = (size_t)bRow * N + bCol;
          sB[k][n] = B[bIdx];
        } else {
          sB[k][n] = 0.0f;
        }
      }
    }

    // 等待所有线程加载完成
    __syncthreads();

// Tile内矩阵乘，内积累加+寄存器分块
#pragma unroll
    for (int k = 0; k < kTileK; k++) {
      //A数组沿M方向
      float aM[kThreadM];
      //B数组沿N方向
      float bN[kThreadN];
#pragma unroll
      for (int m = 0; m < kThreadM; m++) {
        aM[m] = sA[kThreadM * ty + m][k];
      }

#pragma unroll
      for (int n = 0; n < kThreadN; n++) {
        bN[n] = sB[k][kThreadN * tx + n];
      }
#pragma unroll
      for (int m = 0; m < kThreadM; m++) {
        for (int n = 0; n < kThreadN; n++) {
          acc[m][n] += aM[m] * bN[n];
        }
      }
    }

    // 等待所有线程计算完成，再下一轮覆盖共享内存
    __syncthreads();
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

void V6MatMulRegisterTiled(const float *dA, const float *dB, float *dC, int N) {

  constexpr int kBlockN = 32;
  constexpr int kBlockM = 8;
  constexpr int kThreadM = 4;
  constexpr int kThreadN = 4;
  constexpr int kTileK = 64;

  constexpr int kTileM = kBlockM * kThreadM;
  constexpr int kTileN = kBlockN * kThreadN;

  dim3 block(kBlockN, kBlockM);
  dim3 grid((N + kTileN - 1) / kTileN, (N + kTileM - 1) / kTileM);

  MatMulRegisterTiled<kBlockN, kBlockM, kThreadM, kThreadN, kTileK, kTileM,
                      kTileN><<<grid, block>>>(dA, dB, dC, N);
}
