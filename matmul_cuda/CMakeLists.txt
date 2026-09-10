cmake_minimum_required(VERSION 3.24)

# 默认构建类型设为 Release（单配置生成器生效）
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 提前设置 CUDA 架构，project 阶段生效
set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA target SM architecture" FORCE)


project(cuda-matmul-stepwise-optimization LANGUAGES C CXX CUDA)

# ========== 全局强制UTF-8编码，彻底解决C4819警告 ==========
# C/C++ 编译器直接启用 UTF-8
add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
# CUDA 编译时给主机编译器传递 UTF-8 参数
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)

# C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 输出目录：build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ===== 新增：找到 CUDA 工具包（包含 cuBLAS）=====
find_package(CUDAToolkit REQUIRED)

# -------------------------- 编译选项 --------------------------
add_executable(matmul_bench
    benchmark.cu
    kernels/cublas_gemm.cu          # ===== 新增：cuBLAS 封装 =====
    kernels/naive_matmul.cu
    kernels/v1_naive_matmul.cu
    kernels/v2_matmul_basic_tiled.cu
    kernels/v3_matmul_basic_tiled_K64.cu
    kernels/v4_matmul_basic_tiled_K128.cu
    kernels/v5_matmul_register_tiled.cu
    kernels/v6_matmul_register_tiled_opt.cu
    kernels/v7_matmul_register_tiled_opt.cu
    kernels/v8_matmul_double_buffering.cu
    kernels/v9_matmul_float4.cu
    kernels/v10_matmul_wmma.cu
    kernels/v11_matmul_wmma_opt.cu
    kernels/v12_matmul.cu
    kernels/v13_matmul.cu
)

target_compile_options(matmul_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

target_include_directories(matmul_bench PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/kernels
)

# ===== 新增：链接 cuBLAS =====
target_link_libraries(matmul_bench PRIVATE CUDA::cublas)
