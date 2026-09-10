#include "list_matmul.cuh"

// ============================================================================
//  最朴素的 CUDA kernel。
// ============================================================================
__global__ void naiveMatMulKernel(const float* A, const float* B, float* C, int N) {
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

void cuda_naive_matmul(const float* d_A, const float* d_B, float* d_C, int N) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);
    naiveMatMulKernel<<<grid, block>>>(d_A, d_B, d_C, N);
}
