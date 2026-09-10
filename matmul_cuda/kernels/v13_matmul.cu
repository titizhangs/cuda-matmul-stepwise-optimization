#include "list_matmul.cuh"

// ============================================================================
// Tiled: 综合极致优化
// ============================================================================
template <int kThread, int kThreads, int kThreadCoarsening, int kTileK,
          int kBlockOutMN>
static __global__ void MatMul(const float *__restrict__ A,
                              const float *__restrict__ B,
                              float *__restrict__ C, int N) {

  // __shared__ float sA0[kBlockOutMN][kTileK];
  // __shared__ float sB0[kTileK][kBlockOutMN];

  extern __shared__ float smBlock[];

  float *__restrict__ baseA = (float *)__builtin_assume_aligned(smBlock, 16);
  float *__restrict__ baseB =
      (float *)__builtin_assume_aligned(smBlock + kBlockOutMN * kTileK, 16);

  float (&sA0)[kBlockOutMN][kTileK] =
      *reinterpret_cast<float (*)[kBlockOutMN][kTileK]>(baseA);
  float (&sB0)[kTileK][kBlockOutMN] =
      *reinterpret_cast<float (*)[kTileK][kBlockOutMN]>(baseB);

  // 8*8 输出计算坐标划分
  int row = threadIdx.x / kThread;
  int col = threadIdx.x % kThread;

  // 寄存器累加
  float acc[kThreadCoarsening][kThreadCoarsening] = {0.0f};
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

    __syncthreads();

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

    __syncthreads();
  }

  for (int m = 0; m < kThreadCoarsening; m++) {
    for (int n = 0; n < kThreadCoarsening / 4; n++) {
      float4 cVec = *reinterpret_cast<const float4 *>(&acc[m][n * 4]);
      reinterpret_cast<float4 *>(
          &C[(blockIdx.y * kBlockOutMN + row * kThreadCoarsening + m) * N +
             blockIdx.x * kBlockOutMN + col * kThreadCoarsening + n * 4])[0] =
          cVec;
    }
  }
}

void V13MatMul(const float *dA, const float *dB, float *dC, int N) {

  constexpr int kThread = 16;
  constexpr int kThreadCoarsening =
      8; // 限制4的倍数，后面的有float4去写C，不是4的倍数kernel就要写额外的边界条件处理。
  constexpr int kThreads = kThread * kThread;

  // 输出元素128*128，就要加载对应128行的A矩阵数据和128列的B矩阵数据
  constexpr int kBlockOutMN = kThread * kThreadCoarsening; // 128

  // 32*4=128 正好缓存行
  // 8个线程正好覆盖一行，8个float4正好32
  // 2^8线程，其中2^3用来作为k方向坐标，剩下的2^5用来作为M/N方向坐标 128/(2^5)=
  // 4轮循环加载完tile
  constexpr int kTileK = 32;

  dim3 block(kThreads);
  dim3 grid((N + kBlockOutMN - 1) / kBlockOutMN,
            (N + kBlockOutMN - 1) / kBlockOutMN);

  constexpr int sizeA = kBlockOutMN * kTileK;
  constexpr int sizeB = kTileK * kBlockOutMN;
  constexpr int totalSharedBytes = (sizeA + sizeB) * sizeof(float);
  cudaError_t err = cudaFuncSetAttribute(
      reinterpret_cast<const void *>(
          MatMul<kThread, kThreads, kThreadCoarsening, kTileK, kBlockOutMN>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, 101376);

  MatMul<kThread, kThreads, kThreadCoarsening, kTileK, kBlockOutMN>
      <<<grid, block, totalSharedBytes>>>(dA, dB, dC, N);
}
