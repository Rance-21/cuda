#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

// 研究GPU矩阵转置算法
/*
只有形如 if (blockIdx.x > 0) 这种在 Block 尺度上要么全进、要么全不进的分支，
内部放置 __syncthreads() 才是合法安全的
单个warp内__syncthreads()导致dead lock
Volta架构及以后可以，每个thread有单独的PC，靠SMSP的广播掩码实现，但是有L1 i-cache的开销
*/
#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = (call);                                                                                      \
        if (err != cudaSuccess) {                                                                                      \
            std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__);             \
            std::exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                              \
    } while (0)

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    // 当前线程负责的数组位置。
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void block_sum_kernel(const float *input, float *output, int n) {
    extern __shared__ float s[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // 每个线程先把自己的数据搬进 shared memory
    s[tid] = (i < n) ? input[i] : 0.0f;
    // 必须等整个 block 都写完
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s[tid] += s[tid + stride];
        }

        // 必须等这一轮所有加法结束，
        // 下一轮才能读取结果
        __syncthreads();
    }

    // block 的最终结果
    if (tid == 0) {
        output[blockIdx.x] = s[0];
    }
}

void block_sum(const float *h_input, float *h_output, int n) {
    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;

    float *d_input;
    float *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, blocks * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_output, blocks * sizeof(float)));

    // 第三个参数是shared_memory_bytes
    block_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_output, n);

    CUDA_CHECK(cudaMemcpy(h_output, d_output, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

void vector_add() {
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

    for (int i = 0; i < REPEAT; ++i) {
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

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::printf("请正确传参！");
        return 0;
    }

    char choice = *argv[1];
    switch (choice) {
    case 1: {
        vector_add();
    }
    case 2: {
        float *h_input;
        for (int i = 0; i < 1024; ++i) {
            h_input[i] = static_cast<float>(i + 1);
        }
        float *h_output;
        block_sum(h_input, h_output, 1024);

        for (int i = 0; i < 4; i++) {
            std::printf("%f", h_output[i]);
        }
    }
    }
}