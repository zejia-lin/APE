#include "ape.h"
#include "test_gemm.h"

int main(int argc, char **argv) {
    if(argc == 4){
        size_t m = atoi(argv[1]);
        size_t n = atoi(argv[2]);
        size_t k = atoi(argv[3]);
        ape::test_gemm_fp32(m, n, k, ape::APE_ALGO_FP32B);
        ape::test_gemm_fp32(m, n, k, ape::APE_ALGO_AUTO_STRICT);
        return 0;
    }
    for(size_t N = 1024; N < 32768; N *= 2){
        ape::test_gemm_fp32(N, N, N, ape::APE_ALGO_FP32B);
        ape::test_gemm_fp32(N, N, N, ape::APE_ALGO_AUTO_STRICT);
        size_t zhong = N * 1.25;
        ape::test_gemm_fp32(zhong, zhong, zhong, ape::APE_ALGO_FP32B);
        ape::test_gemm_fp32(zhong, zhong, zhong, ape::APE_ALGO_AUTO_STRICT);   
        zhong = N * 1.5;
        ape::test_gemm_fp32(zhong, zhong, zhong, ape::APE_ALGO_FP32B);
        ape::test_gemm_fp32(zhong, zhong, zhong, ape::APE_ALGO_AUTO_STRICT);
    }
}