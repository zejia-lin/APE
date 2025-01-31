#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>

#include <cuda_runtime.h>

namespace ape {

enum ApeTrans {
    APE_TRANS_N = 0,
    APE_TRANS_T,
};

enum ApeAlgo {
    APE_ALGO_AUTO = 1,
    APE_ALGO_AUTO_STRICT,
    APE_ALGO_CUBLAS,
    APE_ALGO_FP32F,
    APE_ALGO_FP32B,
    APE_ALGO_FP32T,
    APE_ALGO_INT16,
};

inline std::string getApeAlgoName(ApeAlgo algo) {
    switch (algo) {
    case APE_ALGO_AUTO:
        return "AUTO";
    case APE_ALGO_AUTO_STRICT:
        return "AUTO_STRICT";
    case APE_ALGO_CUBLAS:
        return "CUBLAS";
    case APE_ALGO_FP32F:
        return "FP32F";
    case APE_ALGO_FP32B:
        return "FP32B";
    case APE_ALGO_FP32T:
        return "FP32T";
    case APE_ALGO_INT16:
        return "INT16";
    default:
        return "Invalid";
    }
}

class APEHandler;

void apeInit(APEHandler *apeHandle, const size_t buf_size = 0, cudaStream_t stream = 0);

void apeDestroy(APEHandler *apeHandle);

void apeGemmFP32(APEHandler apeHandle, ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                 const float *B, int ldb, const float *beta, float *C, int ldc, const ApeAlgo algo = APE_ALGO_AUTO);

void apeGemmFP64(APEHandler apeHandle, ApeTrans transa, ApeTrans transb, int m, int n, int k, const double *alpha, const double *A, int lda,
                 const double *B, int ldb, const double *beta, double *C, int ldc, ApeAlgo algo = APE_ALGO_AUTO);

void apeGemmINT16(APEHandler apeHandle, ApeTrans transa, ApeTrans transb, int m, int n, int k, const int16_t *alpha, const int16_t *A, int lda,
                  const int16_t *B, int ldb, const int32_t *beta, int32_t *C, int ldc, ApeAlgo algo = APE_ALGO_AUTO);

} // namespace ape