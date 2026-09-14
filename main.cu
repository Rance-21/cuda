#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

// 研究GPU矩阵转置算法

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

int main() {
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