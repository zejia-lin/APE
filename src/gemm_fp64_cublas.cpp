#include "common.h"
#include "kernel.h"

namespace ape {
void gemm_fp64_cublas(ApeTrans transa, ApeTrans transb, int m, int n, int k, const double *alpha, const double *A, int lda,
                      const double *B, int ldb, const double *beta, double *C, int ldc) {
    cublasSafeCall(cublasDgemm(APEHandler::getInstance()->getCublasHandle(), cublasOperation_t(transa), cublasOperation_t(transb), m, n, k,
                               alpha, A, lda, B, ldb, beta, C, ldc));
}

} // namespace ape
