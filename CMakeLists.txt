cmake_minimum_required(VERSION 3.24)

project(cuda_learning LANGUAGES CXX CUDA)

add_executable(main main.cu)

set_target_properties(main PROPERTIES
    CUDA_ARCHITECTURES native
)

target_compile_features(main PRIVATE cxx_std_17)

# 保留源码行号，方便 ncu / sanitizer 定位代码，
# 但不像 -G 那样严重影响优化。
target_compile_options(main PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)