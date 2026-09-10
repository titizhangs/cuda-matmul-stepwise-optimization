#include "list_matmul.cuh"
#include <cstdio>

// ============================================================================
// Tiled: 综合极致优化
// ============================================================================
template <int kBlockMN,int kThreadCoarsening,int kTileK,int kTileOutMN>
static __global__ void MatMul(const float *__restrict__ A,
                                             const float *__restrict__ B,
                                             float *__restrict__ C, int N) {

  __shared__ float sA0[kTileOutMN][kTileK];
  __shared__ float sB0[kTileK][kTileOutMN];

  // __shared__ float sA1[kTileOutMN][kTileK];
  // __shared__ float sB1[kTileK][kTileOutMN];


  //寄存器累加
  float acc[kTileOutMN][kTileOutMN]={0.0f};
  // int cur = 0;
  // int next = 1;
  for(int tileK=0;tileK<(N+kTileK-1)/kTileK;tileK++){
    for(int k=0;k<(kTileK+kBlockMN-1)/kBlockMN;k++){
      //by thread (0,8,0) in block (6,0,0)
      //sA0[32-36][]
      for(int coarsening=0;coarsening<kThreadCoarsening;coarsening++){
        //规律：共享内存是单block内共享的，又共享内存分块是以kTileK为K方向分块的，所以共享内存坐标为全局坐标去除含有block坐标，共享内存数组维度大小因子的。
        sA0[threadIdx.y*kThreadCoarsening+coarsening]
        [k*kBlockMN+threadIdx.x]
        =A[(blockIdx.y*kTileOutMN+threadIdx.y*kThreadCoarsening+coarsening)*N
          +tileK*kTileK+k*kBlockMN+threadIdx.x];

        sB0[k*kBlockMN+threadIdx.y][threadIdx.x*kThreadCoarsening+coarsening]
        =B[(tileK*kTileK+k*kBlockMN+threadIdx.y)*N
          +blockIdx.x*kTileOutMN+threadIdx.x*kThreadCoarsening+coarsening];      
      }
    }

    //打印共享内存查问题------START
    CUDA_DEBUG_BLOCK_IF(blockIdx.x == 0 && threadIdx.x==0 && blockIdx.y == 0 && threadIdx.y==0){
      printf("sA:\n");
      for(int m=0;m<kTileOutMN;m++){
        for(int k=0;k<kTileK;k++){
          printf("%8.3f|",sA0[m][k]);
        }
        printf("\n");
      }
      printf("sB:\n");
      for(int k=0;k<kTileK;k++){
        for(int n=0;n<kTileOutMN;n++){
          printf("%8.3f|",sB0[k][n]);
        }
        printf("\n");
      }
    }

    __syncthreads();

    float tmpA[kThreadCoarsening][kTileK]={0.0f};
    float tmpB[kTileK][kThreadCoarsening]={0.0f};
    for(int m=0;m<kThreadCoarsening;m++){
      for(int k=0;k<kTileK;k++){
        tmpA[m][k]=sA0[threadIdx.y*kThreadCoarsening+m][k];
      }
    }
    for(int k=0;k<kTileK;k++){
      for(int n=0;n<kThreadCoarsening;n++){
        tmpB[k][n]=sB0[k][threadIdx.x*kThreadCoarsening+n];
      }
    }
    for(int m=0;m<kThreadCoarsening;m++){
      for(int n=0;n<kThreadCoarsening;n++){
        for(int k=0;k<kTileK;k++){
          acc[m][n]+=tmpA[m][k]*tmpB[k][n];
        }
      }
    }
    __syncthreads();
  }
  for (int m = 0; m < kThreadCoarsening; m++) {
    for (int n = 0; n < kThreadCoarsening; n++) {
      C[(blockIdx.y*kTileOutMN+threadIdx.y*kThreadCoarsening+m)*N
        +blockIdx.x*kTileOutMN+threadIdx.x*kThreadCoarsening+n] = acc[m][n];
    }
  }
}

void V12MatMul(const float *dA, const float *dB, float *dC,
                             int N) {

  constexpr int kBlockMN = 16;
  constexpr int kThreadCoarsening = 4;

  constexpr int kTileOutMN = kBlockMN * kThreadCoarsening;

  constexpr int kTileK = kTileOutMN * 1; //kTileK小于kTileOutMN会造成线程浪费还要加上对这个的边界校验。

  //有线程粗化了，也就不用扁平x维度到32了， 在A和B形状大小相同的前提下x和y相同加载总量最小。
  dim3 block(kBlockMN,kBlockMN );
  dim3 grid((N+kBlockMN-1)/kTileOutMN,(N+kBlockMN-1)/kTileOutMN);

  MatMul<kBlockMN, kThreadCoarsening, kTileK, kTileOutMN><<<grid, block>>>(dA, dB, dC, N);
}
