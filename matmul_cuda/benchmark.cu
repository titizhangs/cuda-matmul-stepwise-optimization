// ============================================================================
//  benchmark.cu : CUDA naive matmul 最简测试 / 压测
// ----------------------------------------------------------------------------
//  固定数据, 不随机: A、B 全部填 1.0f -> 理论 C[i][j] = N (确定性, 每次跑都一样)。
//  用法:  matmul_cuda_bench [-n N]    (N 默认 4096)
//    跑完只打印: 耗时 / GFLOPS / 正确性。
// ============================================================================
#include "kernels/list_matmul.cuh"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include <random>


#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s:%d -> %s\n",                    \
                         __FILE__, __LINE__, cudaGetErrorString(_e));          \
            return 1;                                                          \
        }                                                                      \
    } while (0)


// 统一launch函数签名
using MatmulLaunchFn = void (*)(const float* dA,const float* dB, float* dC, int N);

// 内核元信息：id、显示名字、函数指针
struct KernelEntry {
    int kernel_id;
    const char* desc;
    MatmulLaunchFn launch_fn;
};

static const KernelEntry kernel_table[] = {
    {99, "CublasGemm", CublasGemm } ,
    {0, "cuda_naive_matmul", cuda_naive_matmul},
    {1, "v1_naive_matmul", v1_naive_matmul},
    {2, "v2_matmul_basic_tiled", v2_matmul_basic_tiled },
    {3, "v3_matmul_basic_tiled", v3_matmul_basic_tiled_K64 },
    {4, "v4_matmul_basic_tiled", v4_matmul_basic_tiled_K128 },
    {5, "v5MatmulRegisterTiled", V5MatMulRegisterTiled },
    {6, "v6MatmulRegisterTiled", V6MatMulRegisterTiled },
    {7, "v7MatmulRegisterTiled", V7MatMulRegisterTiled },
    {8, "v8MatmulRegisterTiled", V8MatMulDoubleBuffering },
    {9, "V9MatMulFloat4", V9MatMulFloat4 },
    {10, "V10MatMulWMMA", V10MatMulWMMA },
    {11, "V11MatMulWMMAOpt", V11MatMulWMMAOpt },
    {12, "V12MatMul", V12MatMul },
    {13, "V13MatMul", V13MatMul }
    
};

static const int kernel_table_size = sizeof(kernel_table) / sizeof(kernel_table[0]);


static void fill_random(float* v, unsigned seed, int elems) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for(int i=0;i<elems;i++) v[i]=dist(gen);
}

MatmulLaunchFn run_selected_kernel(int kernel_id){
    for (int i = 0; i < kernel_table_size; ++i)
    {
        const auto& entry = kernel_table[i];
        if (entry.kernel_id == kernel_id)
        {
            printf("Run kernel %d: %s\n", entry.kernel_id, entry.desc);
            return entry.launch_fn;
        }
    }
    printf("[error] unknown kernel_id = %d\n", kernel_id);
    return NULL;
}

// 独立参考: 只算单个 (i,j) 元素, 用于正确性对拍。
static float ref_element(const float* A,
                         const float* B,
                         int N, int i, int j) {
    float s = 0.0f;
    for (int k = 0; k < N; ++k)
        s += A[(size_t)i * N + k] * B[(size_t)k * N + j];
    return s;
}

int main(int argc, char** argv) {
    int N = 4096;
    int kernel_id=0;
    unsigned seed = 12345;
    bool allCheck=false;
    //默认预热1次
    int warm_round=1;
    //默认跑3次
    int run_round=3;

    for (int a = 1; a < argc; ++a) {
        if(std::string(argv[a]) == "--kernel" && a + 1 < argc){
            kernel_id = std::atoi(argv[++a]);
        }else if(std::string(argv[a]) == "-n" && a + 1 < argc){
            N = std::atoi(argv[++a]); 
        }else if(std::string(argv[a]) == "-s" && a + 1 < argc){
            seed = (unsigned)std::atoi(argv[++a]);
        }else if(std::string(argv[a]) == "--allCheck" && a + 1 < argc){
            std::string sAllCheck = argv[++a];
            allCheck=sAllCheck.size()==3 && 
                     std::tolower((unsigned char)sAllCheck[0])=='y' &&
                     std::tolower((unsigned char)sAllCheck[1])=='e' &&
                     std::tolower((unsigned char)sAllCheck[2])=='s';
        }else if(std::string(argv[a]) == "--runRound" && a + 1 < argc){     
            run_round = std::atoi(argv[++a]);        
        }else if(std::string(argv[a]) == "--warmRound" && a + 1 < argc){     
            warm_round = std::atoi(argv[++a]);        
        }
    }
    if (N <= 0) { std::fprintf(stderr, "N must be > 0\n"); return 1; }

    const size_t elems = (size_t)N * N;
    const size_t bytes = elems * sizeof(float);

    // 固定数据: A、B 全 1.0f
    float* h_A = (float*)std::malloc(bytes);
    float* h_B = (float*)std::malloc(bytes);
    float* h_C = (float*)std::malloc(bytes);

    fill_random(h_A, seed, elems);
    fill_random(h_B, seed + 1, elems);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    MatmulLaunchFn kernel=run_selected_kernel(kernel_id);
    if(kernel==NULL) return -1;
    // warm-up (含 context 初始化, 不计入计时)
    for(int i=0;i<warm_round;i++){
       kernel(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float best_ms = 1e30f;
    float sum_ms= 0.0f;
    for (int r = 0; r < run_round; ++r) {
        CUDA_CHECK(cudaEventRecord(start));
        kernel(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        best_ms = (ms < best_ms) ? ms : best_ms;
        sum_ms += ms;
    }
    float avg_ms=sum_ms/run_round;

    double sec = best_ms / 1000.0;
    double best_gflops = (2.0 * (double)N * N * N) / (sec * 1e9);

    double avg_sec = avg_ms / 1000.0;
    double avg_gflops = (2.0 * (double)N * N * N) / (avg_sec * 1e9);

    // 拷回并抽样对拍
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    if(allCheck){
        float* expect=(float*)std::malloc(bytes);;
        printf("expect|got\n");
        for (int m = 0; m < N; ++m) {
            for(int n=0;n<N;++n){
                for(int k=0;k<N;k++){
                    expect[m*N+n]+=h_A[m*N+k]*h_B[k*N+n];
                }
                printf("%8.3f(%8.3f)|", expect[m*N+n],h_C[m*N+n]);
                printf("\n");
            }
        }
        
    } else {
        std::mt19937 chk(seed + 99);
        std::uniform_int_distribution<int> idx(0, N - 1);
        int checks = (N * N < 64) ? (int)(N * N) : 64;
        float max_err = 0.0f, max_val = 0.0f;
        for (int c = 0; c < checks; ++c) {
            int i = idx(chk), j = idx(chk);
            float ref = ref_element(h_A, h_B, N, i, j);
            float got = h_C[(size_t)i * N + j];
            max_err = std::max(max_err, std::fabs(ref - got));
            max_val = std::max(max_val, std::fabs(ref));
        }
        float rel = (max_val > 0.0f) ? (max_err / max_val) : max_err;

        std::printf("N=%d seed=%u  best=%.3f ms  %.2f GFLOPS\n", N, seed, best_ms, best_gflops);
        std::printf("avg=%.3f ms  %.2f GFLOPS\n", avg_ms, avg_gflops);
        std::printf("correctness: sampled %d elems, max_abs_err=%.3e, rel_err=%.3e  %s\n",
                    checks, max_err, rel, (rel < 1e-3f) ? "OK" : "MISMATCH");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    std::free(h_A); std::free(h_B); std::free(h_C);
    return 0;
}
