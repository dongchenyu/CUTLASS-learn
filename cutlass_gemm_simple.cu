#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"

using namespace std;

int main(){
    using ElementA = float;
    using ElementB = float;
    using ElementC = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementA,
        LayoutA,
        ElementB,
        LayoutB,
        ElementC,
        LayoutC
    >;

    int M = 4;
    int N = 4;
    int K = 4;

    cutlass::HostTensor<ElementA, LayoutA> A({M, K});
    cutlass::HostTensor<ElementB, LayoutB> B({K, N});
    cutlass::HostTensor<ElementC, LayoutC> C({M, N});
    cutlass::HostTensor<ElementC, LayoutC> D({M, N});

    for(int i = 0; i < A.capacity(); i ++){
        A.host_data()[i] = 1.0f;
    }

    for(int i = 0; i < B.capacity(); i ++){
        B.host_data()[i] = 1.0f;
    }

    for(int i = 0; i < C.capacity(); i ++){
        C.host_data()[i] = 1.0f;
    }

    A.sync_device();
    B.sync_device();
    C.sync_device();

    float alpha = 1.0f;
    float beta = 0.0f;

    typename Gemm::Arguments arguments(
        {M, N, K},
        {A.device_data(), A.stride(0)},
        {B.device_data(), B.stride(0)},
        {C.device_data(), C.stride(0)},
        {D.device_data(), D.stride(0)},
        {alpha, beta}
    );

    Gemm gemm_operator;
    
    cutlass::Status status = gemm_operator(arguments);

    if(status != cutlass::Status::kSuccess){
        cout << "CUTLASS GEMM failed" << endl;
        return -1;
    }
    D.sync_host();
    cout << "D = " << endl;
    
    for(int i = 0; i < M; i ++){
        for(int j = 0; j < N; j ++){
            cout << D.at({i, j}) << " ";
        }
        cout << endl;
    }
    return 0;
}

/*
nvcc cutlass_gemm_simple.cu \
  -I ../../include \
  -I ../../tools/util/include \
  -arch=sm_70 \
  -std=c++17 \
  -O3 \
  -o cutlass_gemm_simple
*/