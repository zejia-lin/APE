#pragma once

#include "ape.h"
#include "common.h"

namespace ape {

#ifdef ARCH_SM80
constexpr int NUM_SM = 108;
constexpr int MAX_THREAD = 1024;
#endif
#ifdef ARCH_SM70
constexpr int NUM_SM = 80;
constexpr int MAX_THREAD = 1024;
#endif

constexpr int AUTO_BLOCK = 128;

constexpr float FP32F_MAX = 65504.0f;
constexpr float FP32F_MIN = 3.1e-5f;
constexpr float FP32B_MAX = 3.38e38f;
constexpr float FP32B_MIN = 3.9e-34f;
constexpr int INT16C_MAX = 32639;

void gemm_fp32_auto(APEHandler apeHandle, ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                    const float *B, int ldb, const float *beta, float *C, int ldc);
void gemm_fp32_auto_strict(APEHandler apeHandle, ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                           const float *B, int ldb, const float *beta, float *C, int ldc);
void gemm_fp32_cublas(APEHandler apeHandle, ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                      const float *B, int ldb, const float *beta, float *C, int ldc);
void gemm_fp32_fp32f(APEHandler apeHandle, ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                     const float *B, int ldb, const float *beta, float *C, int ldc);
void gemm_fp32_fp32b(APEHandler apeHandle, ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                     const float *B, int ldb, const float *beta, float *C, int ldc);
void gemm_fp32_fp32t(APEHandler apeHandle, ApeTrans transa, ApeTrans transb, int m, int n, int k, const float *alpha, const float *A, int lda,
                     const float *B, int ldb, const float *beta, float *C, int ldc);
void gemm_fp64_cublas(APEHandler apeHandle, ApeTrans transa, ApeTrans transb, int m, int n, int k, const double *alpha, const double *A, int lda,
                      const double *B, int ldb, const double *beta, double *C, int ldc);
void gemm_int16_emu(APEHandler apeHandle, ApeTrans transa, ApeTrans transb, int m, int n, int k, const int16_t *alpha, const int16_t *A, int lda,
                    const int16_t *B, int ldb, const int32_t *beta, int32_t *C, int ldc);

void convert_fp64_to_fp32(float *dst, const double *src, size_t size);
void convert_fp32_to_fp64(double *dst, const float *src, size_t size);
void convert_int32_to_int16(int16_t *dst, const int32_t *src, size_t size);
void convert_int16_to_int32(int32_t *dst, const int16_t *src, size_t size);
void compare_fp32_to_fp64(const float *src, const double *dst, size_t size, double &max_error, double &mean_error);

void split_fp32_to_fp16(half *dst, const float *src, size_t size);
void merge_fp16_to_fp32(float *dst, const half *src, size_t size);
void split_fp32_to_bf16(__nv_bfloat16 *dst, const float *src, uint32_t size);
void merge_bf16_to_fp32(float *dst, const __nv_bfloat16 *src, uint32_t size);
void split_fp32_to_tf32(float *dst, const float *src, size_t size);
void merge_tf32_to_fp32(float *dst, const float *src, size_t size);
void split_int16_to_int8(int8_t *dst, const int16_t *src, size_t size);
void merge_int8_to_int16(int16_t *dst, const int8_t *src, size_t size);
void create_mask_fp32(const float *src, size_t row, size_t col, ApeTrans trans, int8_t *mask);
int count_overflow_fp32f(const float *src, size_t row, size_t col);
int count_overflow_fp32f_strict(const float *src, size_t row, size_t col);
int count_overflow_int16emu(const int16_t *src, size_t row, size_t col);

__device__ inline double fmax(double a, double b) { return (a > b) ? a : b; }
__device__ inline double fabs(double a) { return (a > 0) ? a : -a; }

} // namespace ape