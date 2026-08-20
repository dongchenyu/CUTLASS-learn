#include<cuda_runtime.h>
#include<iostream>
#include<vector>
#include<cmath>

using namespace std;

#define CHECK_CUDA(call)                                                \
do {                                                                    \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        cerr << "CUDA Error: " << cudaGetErrorString(err)           \
                  << " at " << __FILE__ << ":" << __LINE__ << endl;\
        exit(1);                                                        \
    }                                                                   \
} while (0)

void cpu_gemm(
    const vector<float>& A,
    const vector<float>& B,
    vector<float>& C,
    int M, int N, int K){
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

__global__ void gemm_block_warp_thread(const float* A, const float* B, float* C, int M, int N, int K){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;

    int WM = 32;
    int WN = 32;
    int BM = 64;
    int BN = 64;

    int cta_row = BM * by;
    int cta_col = BN * bx;

    int warpId = tid / 32;
    int laneId = tid % 32;

    int warp_m = warpId / 2;
    int warp_n = warpId % 2;

    int warp_row = cta_row + WM * warp_m;
    int warp_col = cta_col + WN * warp_n;

    float acc[32] = {0.0f};

    for(int k = 0; k < K; k ++){
        for(int i = laneId; i < WM * WN; i += 32){
            int local_m = i / WN;
            int local_n = i % WN;
            float a = A[(warp_row + local_m) * K + k];
            float b = B[k * N + (warp_col + local_n)];
            acc[i / 32] += a * b;
        }
    }

    for(int i = laneId; i < WM * WN; i += 32){
        int local_m = i / WN;
        int local_n = i % WN;
        C[(warp_row + local_m) * M + (warp_col + local_n)] = acc[i / 32];
    }

}

__global__ void gemm_shared(const float* A, const float* B, float* C, int M, int N, int K){
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 16;

    constexpr int WM = 32;
    constexpr int WN = 32;

    int tid = threadIdx.x;

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int cta_row = blockIdx.y * BM;
    int cta_col = blockIdx.x * BN;

    int warp_m = warp_id / 2;
    int warp_n = warp_id % 2;

    int warp_row = warp_m * WM;
    int warp_col = warp_n * WN;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float acc[32] = {0.0f};

    for(int bk = 0; bk < K; bk += BK){
        int sa_row = tid / 2;
        int sa_col = (tid % 2) * 8;
        int sb_row = tid / 8;
        int sb_col = (tid % 8) * 8;
        for(int i = 0; i < 8; i ++){
            As[sa_row][sa_col + i] = A[(cta_row + sa_row) * K + bk + sa_col + i];
            Bs[sb_row][sb_col + i] = B[(sb_row + bk) * N + cta_col + sb_col + i];
            // iterator_A.load(tb_frag_A);
            // smem_iterator_A_.store(tb_frag_A);
        }
        __syncthreads();

        for(int k = 0; k < BK; k ++){
            for(int i = lane_id; i < WM * WN; i += 32){
                int local_m = i / WN;
                int local_n = i % WN;
                int smem_m = warp_row + local_m;
                int smem_n = warp_col + local_n;
                //warp_tile_iterator_A_.load(warp_frag_A);
                //warp_tile_iterator_B_.load(warp_frag_B);
                acc[i / 32] += As[smem_m][k] * Bs[k][smem_n];
            }
        }
        __syncthreads();
    }
    for(int i = lane_id; i < WM * WN; i += 32){
        int row = warp_row + cta_row + i / WN;
        int col = warp_col + cta_col + i % WN;
        C[row * M + col] = acc[i / 32];
    }
}

int main(){
    const int M = 128;
    const int N = 128;
    const int K = 128;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    vector<float> h_A(M * K);
    vector<float> h_B(K * N);
    vector<float> h_C(M * N, 0.0f);
    vector<float> h_ref(M * N, 0.0f);

    for(int i = 0; i < M * K; i ++){
        h_A[i] = static_cast<float>((i % 7) - 3) / 7.0f;
    }

    for(int i = 0; i < K * N; i ++){
        h_B[i] = static_cast<float>((i % 5) - 2) / 5.0f;
    }

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    constexpr int BM = 64;
    constexpr int BN = 64;

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(128);

    cout << "grid = (" << grid.x << ", " << grid.y << ")\n";
    cout << "block threads = " << block.x << "\n";

    //gemm_block_warp_thread<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    gemm_shared<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    cpu_gemm(h_A, h_B, h_ref, M, N, K);

    float max_diff = 0.0f;
    int max_idx = -1;

    for (int i = 0; i < M * N; ++i) {
        float diff = fabs(h_C[i] - h_ref[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }

    cout << "max diff = " << max_diff << "\n";

    if (max_diff < 1e-4f) {
        cout << "Result: PASS\n";
    }
    else {
        cout << "Result: FAIL\n";
        if (max_idx >= 0) {
            int row = max_idx / N;
            int col = max_idx % N;
            cout << "Mismatch at (" << row << ", " << col << ")\n";
            cout << "GPU = " << h_C[max_idx] << ", CPU = " << h_ref[max_idx] << "\n";
        }
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}