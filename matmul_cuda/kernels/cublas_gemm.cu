#include "list_matmul.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
// 封装 cublasSgemm，接口和你自己的 kernel 保持一致
// 计算 C = A × B，三个矩阵都是 N×N 的行主序（row-major）
void CublasGemm(const float *dA, const float *dB, float *dC, int N) {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
        // 强制只用 CUDA Core，不用 Tensor Core
        cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // ================================================================
    // 关键：行主序 → 列主序的参数转换
    //
    // cuBLAS 是列主序（column-major），我们的矩阵是行主序（row-major）。
    // 利用性质：行主序的 A(M×K) 在内存布局上 = 列主序的 A^T(K×M)
    //
    // 我们要算：C(M×N) = A(M×K) × B(K×N)  （行主序）
    // 等价于：  C^T(N×M) = B^T(N×K) × A^T(K×M)  （列主序）
    //
    // 所以 cuBLAS 的参数：
    //   第一个矩阵 = B（内存中即 B^T 的列主序布局），不转置
    //   第二个矩阵 = A（内存中即 A^T 的列主序布局），不转置
    //   m = N, n = M, k = K
    //   lda = N（B 行主序每行 N 个元素 = B^T 列主序的 leading dimension）
    //   ldb = K（A 行主序每行 K 个元素 = A^T 列主序的 leading dimension）
    //   ldc = N（C 行主序每行 N 个元素 = C^T 列主序的 leading dimension）
    // ================================================================

    cublasSgemm(
        handle,
        CUBLAS_OP_N,   // transa: 第一个矩阵（B^T）不转置
        CUBLAS_OP_N,   // transb: 第二个矩阵（A^T）不转置
        N,             // m: C^T 的行数 = N
        N,             // n: C^T 的列数 = M = N（方阵）
        N,             // k: 公共维度 = K = N（方阵）
        &alpha,
        dB,            // A: cuBLAS 的第一个矩阵 = 我们的 B
        N,             // lda: B 的 leading dimension = N
        dA,            // B: cuBLAS 的第二个矩阵 = 我们的 A
        N,             // ldb: A 的 leading dimension = N
        &beta,
        dC,            // C: 结果矩阵
        N              // ldc: C 的 leading dimension = N
    );
}
