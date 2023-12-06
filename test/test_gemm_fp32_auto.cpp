#include "ape.h"
#include "test_gemm.h"

int main() {
    for(size_t N = 1024; N < 32768; N *= 2){
        ape::test_gemm_fp32(N, N, N, ape::APE_ALGO_AUTO);
        ape::test_gemm_fp32(N, N, N, ape::APE_ALGO_AUTO_STRICT);
        if(N == 16384){return 0;}
        size_t zhong = N * 1.5;
        ape::test_gemm_fp32(zhong, zhong, zhong, ape::APE_ALGO_AUTO);
        ape::test_gemm_fp32(zhong, zhong, zhong, ape::APE_ALGO_AUTO_STRICT);
    }
}