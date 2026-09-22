#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include <cuda_fp16.h>

// 研究GPU矩阵转置算法
/*
只有形如 if (blockIdx.x > 0) 这种在 Block 尺度上要么全进、要么全不进的分支，
内部放置 __syncthreads() 才是合法安全的
单个warp内__syncthreads()导致dead lock
Volta架构及以后可以，每个thread有单独的PC，靠SMSP的广播掩码实现，但是有L1 i-cache的开销
*/

#define CUDA_CHECK(call)                                                                                   \
    do                                                                                                     \
    {                                                                                                      \
        cudaError_t err = (call);                                                                          \
        if (err != cudaSuccess)                                                                            \
        {                                                                                                  \
            std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE);                                                                       \
        }                                                                                                  \
    } while (0)

__global__ void transpose_naive_fp16_1(
    const __half *input,
    __half *output);

__global__ void vector_add(const float *a, const float *b, float *c, int n)
{
    // 当前线程负责的数组位置。
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void block_sum_kernel(const float *input, float *output, int n)
{
    extern __shared__ float s[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // 每个线程先把自己的数据搬进 shared memory
    s[tid] = (i < n) ? input[i] : 0.0f;
    // 必须等整个 block 都写完
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            s[tid] += s[tid + stride];
        }

        // 必须等这一轮所有加法结束，
        // 下一轮才能读取结果
        __syncthreads();
    }

    // block 的最终结果
    if (tid == 0)
    {
        output[blockIdx.x] = s[0];
    }
}

void block_sum(const float *h_input, float *h_output, int n)
{
    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;

    float *d_input;
    float *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_output, blocks * sizeof(float)));

    // 第三个参数是shared_memory_bytes
    block_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_output, n);

    CUDA_CHECK(cudaMemcpy(h_output, d_output, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

void vector_add()
{
    // 强制编译期计算
    constexpr int N = 1 << 24;
    constexpr int THREADS = 256;
    constexpr int REPEAT = 100;

    const size_t bytes = N * sizeof(float);

    // ---------------- CPU 内存 ----------------
    std::vector<float> h_a(N, 1.0f);
    std::vector<float> h_b(N, 2.0f);
    std::vector<float> h_c(N);

    // ---------------- GPU 显存 ----------------
    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // ---------------- kernel 配置 ----------------
    // 向上取整，确保最后不足 256 个元素也有线程处理。
    const int blocks = (N + THREADS - 1) / THREADS;

    // 先预热一次。
    vector_add<<<blocks, THREADS>>>(d_a, d_b, d_c, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------------- GPU 计时 ----------------
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEAT; ++i)
    {
        vector_add<<<blocks, THREADS>>>(d_a, d_b, d_c, N);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    // Cuda事件同步
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / REPEAT;

    // 理想情况下总流量约 12 bytes / element
    const double transferred_bytes = static_cast<double>(N) * sizeof(float) * 3;
    const double bandwidth_gb_s = transferred_bytes / (avg_ms / 1000.0) / 1e9;

    std::printf("Average kernel time: %.3f ms\n", avg_ms);
    std::printf("Effective bandwidth: %.2f GB/s\n", bandwidth_gb_s);

    // ---------------- 检查结果 ----------------
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    std::printf("c[0] = %.1f\n", h_c[0]);
    std::printf("c[N-1] = %.1f\n", h_c[N - 1]);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

__global__ void normal_kernel(const float *x, float *y, int n, float scale, float bias)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
        return;

    float v = x[i];

    v = v * scale + bias;

    if (v < 0.0f)
        v = 0.0f;

    y[i] = v;
}

// 实际差距要大，因为这个内核函数没利用合并访存
__global__ void coarse_kernel(const float *x, float *y, int n, float scale, float bias)
{
    constexpr int ITEMS = 32;

    int start = (blockIdx.x * blockDim.x + threadIdx.x) * ITEMS;

    // 每个线程暂存 32 个元素。
    float v[ITEMS];

#pragma unroll
    for (int j = 0; j < ITEMS; ++j)
    {
        int i = start + j;

        if (i < n)
            v[j] = x[i];
        else
            v[j] = 0.0f;
    }

// 对 32 个元素做融合操作。
#pragma unroll
    for (int j = 0; j < ITEMS; ++j)
    {
        v[j] = v[j] * scale + bias;

        if (v[j] < 0.0f)
            v[j] = 0.0f;
    }

// 最后统一写回。
#pragma unroll
    for (int j = 0; j < ITEMS; ++j)
    {
        int i = start + j;

        if (i < n)
            y[i] = v[j];
    }
}

void run_normal(const float *d_x, float *d_y, int n)
{
    constexpr int threads = 512;

    int blocks = (n + threads - 1) / threads;

    normal_kernel<<<blocks, threads>>>(d_x, d_y, n, 1.1f, -0.2f);
}

void run_coarse(const float *d_x, float *d_y, int n)
{
    constexpr int threads = 512;
    constexpr int items = 32;

    int blocks = (n + threads * items - 1) / (threads * items);

    coarse_kernel<<<blocks, threads>>>(d_x, d_y, n, 1.1f, -0.2f);
}

void run_register_cliff()
{
    constexpr int N = 1 << 25;

    size_t bytes = N * sizeof(float);

    float *d_x;
    float *d_y;

    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);

    cudaMemset(d_x, 0, bytes);

    cudaFuncAttributes normal_attr{};
    cudaFuncAttributes coarse_attr{};

    cudaFuncGetAttributes(&normal_attr, normal_kernel);

    cudaFuncGetAttributes(&coarse_attr, coarse_kernel);

    int normal_blocks;
    int coarse_blocks;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&normal_blocks, normal_kernel, 512, 0);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&coarse_blocks, coarse_kernel, 512, 0);

    std::printf("Normal : %d registers/thread, %d blocks/SM\n", normal_attr.numRegs, normal_blocks);

    std::printf("Coarse : %d registers/thread, %d blocks/SM\n", coarse_attr.numRegs, coarse_blocks);

    run_normal(d_x, d_y, N);
    cudaDeviceSynchronize();

    run_coarse(d_x, d_y, N);
    cudaDeviceSynchronize();

    cudaFree(d_x);
    cudaFree(d_y);
}

void run_transpose_benchmark()
{
    constexpr int N = 4096;

    constexpr std::size_t ELEMENT_COUNT =
        static_cast<std::size_t>(N) * N;

    constexpr std::size_t MATRIX_BYTES =
        ELEMENT_COUNT * sizeof(__half);

    constexpr int WARMUP_COUNT = 10;
    constexpr int TEST_COUNT = 100;

    // 1. 构造 CPU 矩阵
    std::vector<__half> h_input(ELEMENT_COUNT);
    std::vector<__half> h_output(ELEMENT_COUNT);

    for (int row = 0; row < N; ++row)
    {
        for (int col = 0; col < N; ++col)
        {
            // 控制在 FP16 可以精确表示的小整数范围内，
            // 同时让 row/col 都影响数据，方便检查转置是否正确。
            float value =
                static_cast<float>((row * 131 + col * 17) % 2048);

            h_input[static_cast<std::size_t>(row) * N + col] = __float2half(value);
        }
    }

    // 2. 分配 GPU 内存
    __half *d_input = nullptr;
    __half *d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, MATRIX_BYTES));
    CUDA_CHECK(cudaMalloc(&d_output, MATRIX_BYTES));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input.data(),
        MATRIX_BYTES,
        cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    dim3 grid(N / 64, N / 64); // 64 × 64 blocks

    // 4. Warm up
    for (int i = 0; i < WARMUP_COUNT; ++i)
    {
        transpose_naive_fp16_1<<<grid, block>>>(
            d_input,
            d_output);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. CUDA Event 测纯 kernel 时间
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < TEST_COUNT; ++i)
    {
        transpose_naive_fp16_1<<<grid, block>>>(
            d_input,
            d_output);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms,
        start,
        stop));

    float average_ms =
        total_ms / static_cast<float>(TEST_COUNT);

    // 6. 算 effective bandwidth
    double transferred_bytes =
        2.0 * static_cast<double>(MATRIX_BYTES);

    double seconds =
        static_cast<double>(average_ms) * 1e-3;

    double bandwidth_gbps =
        transferred_bytes / seconds / 1e9;

    std::printf(
        "Matrix: %d x %d FP16\n"
        "Kernel time: %.4f ms\n"
        "Effective bandwidth: %.2f GB/s\n",
        N,
        N,
        average_ms,
        bandwidth_gbps);

    // 7. 拷回并验证
    // 不放进计时范围
    CUDA_CHECK(cudaMemcpy(
        h_output.data(),
        d_output,
        MATRIX_BYTES,
        cudaMemcpyDeviceToHost));

    bool correct = true;

    for (int row = 0; row < N && correct; ++row)
    {
        for (int col = 0; col < N; ++col)
        {

            float input_value = __half2float(
                h_input[static_cast<std::size_t>(row) * N + col]);

            float output_value = __half2float(
                h_output[static_cast<std::size_t>(col) * N + row]);

            if (input_value != output_value)
            {
                std::printf(
                    "Mismatch: input[%d][%d] = %.1f, "
                    "output[%d][%d] = %.1f\n",
                    row,
                    col,
                    input_value,
                    col,
                    row,
                    output_value);

                correct = false;
                break;
            }
        }
    }

    std::printf(
        "Correctness: %s\n",
        correct ? "PASS" : "FAIL");
    // 8. 清理
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::printf("请正确传参！");
        return 0;
    }

    char choice = *argv[1];
    switch (choice)
    {
    case '1':
    {
        std::printf("执行向量加法");

        vector_add();

        break;
    }
    case '2':
    {
        std::printf("执行规约(__syncthreads)");

        float h_input[1024];
        for (int i = 0; i < 1024; ++i)
        {
            h_input[i] = static_cast<float>(i + 1);
        }

        float h_output[1024];

        block_sum(h_input, h_output, 1024);

        for (int i = 0; i < 4; i++)
        {
            std::printf("%f ", h_output[i]);
        }

        break;
    }
    case '3':
    {
        std::printf("执行每个线程寄存器过多导致分配不满。性能悬崖");

        run_register_cliff();

        break;
    }
    case '4':
    {
        std::printf("4096 * 4096 的 fp16 转置 ver 1 \n");
        run_transpose_benchmark();
    }
    }
}

// 4096 * 4096 矩阵转置，该显卡 1536 thread / block，会有大的浪费
// dim3 block(32, 32) | dim3 grid(64, 64)
__global__ void transpose_naive_fp16_1(
    const __half *input,
    __half *output)
{
    constexpr int N = 4096;
    // 要处理的两行中的第一行
    int global_row0 = (blockIdx.y * 64 + threadIdx.y);
    int global_row1 = global_row0 + 32;
    int global_col = (blockIdx.x * 64 + threadIdx.x * 2);
    int tile_row0 = threadIdx.y;
    int tile_row1 = tile_row0 + 32;
    int tile_col = threadIdx.x * 2;

    __shared__ __half tile[64][65];

    __half2 fst_two_elm = *reinterpret_cast<const __half2 *>(
        &input[global_row0 * N + global_col]);
    __half2 scd_two_elm = *reinterpret_cast<const __half2 *>(
        &input[global_row1 * N + global_col]);

    // 正好避免 bank 冲突
    tile[tile_row0][tile_col] = __low2half(fst_two_elm);
    tile[tile_row0][tile_col + 1] = __high2half(fst_two_elm);
    tile[tile_row1][tile_col] = __low2half(scd_two_elm);
    tile[tile_row1][tile_col + 1] = __high2half(scd_two_elm);

    __syncthreads();

    // 转置后，这个 warp 负责 output tile 中的两行
    int output_row0 = blockIdx.x * 64 + threadIdx.y * 2;
    int output_row1 = output_row0 + 1;

    // 每个线程负责该行中的两个连续元素
    int output_col = blockIdx.y * 64 + threadIdx.x * 2;

    // output_row0 对应原 tile 的第 2*threadIdx.y 列
    __half2 out0 = __halves2half2(
        tile[threadIdx.x * 2][threadIdx.y * 2],
        tile[threadIdx.x * 2 + 1][threadIdx.y * 2]);

    // output_row1 对应原 tile 的第 2*threadIdx.y+1 列
    __half2 out1 = __halves2half2(
        tile[threadIdx.x * 2][threadIdx.y * 2 + 1],
        tile[threadIdx.x * 2 + 1][threadIdx.y * 2 + 1]);

    *reinterpret_cast<__half2 *>(
        &output[output_row0 * N + output_col]) = out0;

    *reinterpret_cast<__half2 *>(
        &output[output_row1 * N + output_col]) = out1;
}