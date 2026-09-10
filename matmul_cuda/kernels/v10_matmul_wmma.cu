#include "list_matmul.cuh"
#include <mma.h>
using namespace nvcuda;

// ============================================================================
// WMMA 张量核心 GEMM（TF32，16×16×8）
// ============================================================================
template <int kTileM, int kTileN, int kTileK, int kWarpsM, int kWarpsN>
__global__ void MatMulWMMA(const float *__restrict__ A,
                           const float *__restrict__ B, float *__restrict__ C,
                           int N) {
  constexpr int kWmmaM = 16;
  constexpr int kWmmaN = 16;
  constexpr int kWmmaK = 8; // TF32: K=8

  const int warpId = threadIdx.x / 32;
  const int warpM = warpId / kWarpsN;
  const int warpN = warpId % kWarpsN;

  const int gRowBase = blockIdx.y * kTileM + warpM * kWmmaM;
  const int gColBase = blockIdx.x * kTileN + warpN * kWmmaN;

  __shared__ float sA[kTileM][kTileK];
  __shared__ float sB[kTileK][kTileN];

  // fragment 定义
  wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, wmma::precision::tf32,
                 wmma::row_major>
      aFrag;
  wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, wmma::precision::tf32,
                 wmma::row_major>
      bFrag;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> cFrag;
  wmma::fill_fragment(cFrag, 0.0f);

  // K 方向循环
  const int numTilesK = (N + kTileK - 1) / kTileK;
  for (int tileK = 0; tileK < numTilesK; ++tileK) {
    const size_t kBase = (size_t)tileK * kTileK;

    // 加载 A 到共享内存
    for (int i = threadIdx.x; i < kTileM * kTileK; i += blockDim.x) {
      const int row = i / kTileK;
      const int col = i % kTileK;
      sA[row][col] = A[(blockIdx.y * kTileM + row) * N + (kBase + col)];
    }
    // 加载 B 到共享内存
    for (int i = threadIdx.x; i < kTileK * kTileN; i += blockDim.x) {
      const int row = i / kTileN;
      const int col = i % kTileN;
      sB[row][col] = B[(kBase + row) * N + (blockIdx.x * kTileN + col)];
    }
    __syncthreads();

// 每个 warp 加载 fragment 并计算
#pragma unroll
    for (int k = 0; k < kTileK; k += kWmmaK) {
      wmma::load_matrix_sync(aFrag, &sA[warpM * kWmmaM][k], kTileK);
      wmma::load_matrix_sync(bFrag, &sB[k][warpN * kWmmaN], kTileN);
      wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
    }
    __syncthreads();
  }

  // 存储结果
  wmma::store_matrix_sync(&C[gRowBase * N + gColBase], cFrag, N,
                          wmma::mem_row_major);
}

void V10MatMulWMMA(const float *dA, const float *dB, float *dC, int N) {
  constexpr int kWarpsM = 2;
  constexpr int kWarpsN = 4;
  constexpr int kNumWarps = kWarpsM * kWarpsN; // 8 warp = 256 线程
  constexpr int kTileM = kWarpsM * 16;         // 32
  constexpr int kTileN = kWarpsN * 16;         // 64
  constexpr int kTileK = 64;

  dim3 block(32 * kNumWarps); // 256 线程，一维
  dim3 grid((N + kTileN - 1) / kTileN, (N + kTileM - 1) / kTileM);

  MatMulWMMA<kTileM, kTileN, kTileK, kWarpsM, kWarpsN>
      <<<grid, block>>>(dA, dB, dC, N);
}
