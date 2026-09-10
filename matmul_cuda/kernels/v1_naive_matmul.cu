#include "list_matmul.cuh"

// ============================================================================
//  不变block的线程总数的前提下，将blockDim=(16,16) 改成blockDim=(32,8);
//  减半了A矩阵的无效section请求
//  B矩阵的缓存行级利用率从50%提高到100%
// ============================================================================
static __global__ void naiveMatMulKernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[(size_t)row * N + k] * B[(size_t)k * N + col];
        }
        C[(size_t)row * N + col] = sum;
    }
}

void v1_naive_matmul(const float* d_A, const float* d_B, float* d_C, int N) {
    dim3 block(32, 8);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);
    naiveMatMulKernel<<<grid, block>>>(d_A, d_B, d_C, N);
}
