#include <iostream>

#include "common.h"
#include "kernel.h"

namespace ape {
void gemm_fp32_fp32f(ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                     const float *B, int ldb, const float *beta, float *C, int ldc) {
    assert((1ULL * m * k + k * n) * 4 <= APEHandler::getInstance()->getBufSize());
    half *buf = (half *)APEHandler::getInstance()->getBuf();
    half *buf_a, *buf_b;
    buf_a = (half *)APEHandler::getInstance()->getBuf();
    buf_b = (half *)(APEHandler::getInstance()->getBuf()) + m * k * 2;

    split_fp32_to_fp16(buf_a, A, m * k);
    split_fp32_to_fp16(buf_b, B, k * n);

    float alpha0 = *alpha, alpha1 = *alpha / 4096.0f, beta0 = *beta, beta1 = 1;
    cublasSafeCall(cublasGemmEx(APEHandler::getInstance()->getCublasHandle(), cublasOperation_t(transa), cublasOperation_t(transb), m, n, k,
                                &alpha0, buf_a, CUDA_R_16F, lda, buf_b, CUDA_R_16F, ldb, &beta0, C, CUDA_R_32F, ldc,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    cublasSafeCall(cublasGemmEx(APEHandler::getInstance()->getCublasHandle(), cublasOperation_t(transa), cublasOperation_t(transb), m, n, k,
                                &alpha1, buf_a + m * k, CUDA_R_16F, lda, buf_b, CUDA_R_16F, ldb, &beta1, C, CUDA_R_32F, ldc,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    cublasSafeCall(cublasGemmEx(APEHandler::getInstance()->getCublasHandle(), cublasOperation_t(transa), cublasOperation_t(transb), m, n, k,
                                &alpha1, buf_a, CUDA_R_16F, lda, buf_b + k * n, CUDA_R_16F, ldb, &beta1, C, CUDA_R_32F, ldc,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    // cudaSafeCall(cudaFree(buf));
}

} // namespace ape
