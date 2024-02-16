
#include <iomanip>

#include "ape.h"
#include "common.h"
#include "kernel.h"

namespace ape {
void test_gemm_fp32(size_t seed, float alpha, float beta, float mean, float stdev,
                    int m, int n, int k, ape::ApeAlgo algo, int warmups = 2, int iterations = 10);
void test_gemm_fp32(int m, int n, int k, ape::ApeAlgo algo, int warmups = 2, int iterations = 10) {
  test_gemm_fp32(42ULL, 0.23456f, 0.23456f, 0.f, 1.f, m, n, k, algo, warmups, iterations);
}
void test_gemm_fp32(size_t seed, float alpha, float beta, float mean, float stdev,
                    int m, int n, int k, ape::ApeAlgo algo, int warmups, int iterations) {
  int width;
  switch (algo) {
  case APE_ALGO_AUTO:
  case APE_ALGO_AUTO_STRICT:
    width = 8;
    break;
  case APE_ALGO_FP32F:
    width = 4;
    break;
  case APE_ALGO_FP32B:
    width = 6;
    break;
  case APE_ALGO_FP32T:
    width = 8;
    break;
  default:
    width = 0;
  }
  APEHandler apeHandle;
  float *data_eval_a = 0, *data_eval_b = 0, *data_eval_c = 0, *data_backup_c = 0;
  cudaSafeCall(cudaMallocManaged((void **)&data_eval_a, m * k * sizeof(float)));
  cudaSafeCall(cudaMallocManaged((void **)&data_eval_b, k * n * sizeof(float)));
  cudaSafeCall(cudaMallocManaged((void **)&data_eval_c, m * n * sizeof(float)));
  double *data_res_a = 0, *data_res_b = 0, *data_res_c = 0;
  cudaSafeCall(cudaMallocManaged((void **)&data_res_a, m * k * sizeof(double)));
  cudaSafeCall(cudaMallocManaged((void **)&data_res_b, k * n * sizeof(double)));
  cudaSafeCall(cudaMallocManaged((void **)&data_res_c, m * n * sizeof(double)));

  curandGenerator_t gen;
  curandSafeCall(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  curandSafeCall(curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_DEFAULT));
  curandSafeCall(curandSetPseudoRandomGeneratorSeed(gen, seed));
  curandSafeCall(curandGenerateNormal(gen, data_eval_a, m * k, mean, stdev));
  curandSafeCall(curandGenerateNormal(gen, data_eval_b, k * n, mean, stdev));
  curandSafeCall(curandGenerateNormal(gen, data_eval_c, m * n, mean, stdev));

  ape::convert_fp32_to_fp64(data_res_a, data_eval_a, m * k);
  ape::convert_fp32_to_fp64(data_res_b, data_eval_b, k * n);
  ape::convert_fp32_to_fp64(data_res_c, data_eval_c, m * n);

  float alpha_eval = alpha;
  float beta_eval = beta;
  double alpha_res = alpha_eval, beta_res = beta_eval;
  apeInit(&apeHandle, (1ULL * m * k + k * n) * width);
  ape::apeGemmFP64(apeHandle, ape::APE_TRANS_T, ape::APE_TRANS_N, m, n, k, &alpha_res, data_res_a, k, data_res_b, k, &beta_res,
                   data_res_c, m, ape::APE_ALGO_CUBLAS);

  cudaSafeCall(cudaFree(data_res_a));
  cudaSafeCall(cudaFree(data_res_b));
  // cudaSafeCall(cudaFree(data_res_c));

  ape::apeGemmFP32(apeHandle, ape::APE_TRANS_T, ape::APE_TRANS_N, m, n, k, &alpha_eval, data_eval_a, k, data_eval_b, k, &beta_eval,
                   data_eval_c, m, algo);
  double max_error, mean_error;
  ape::compare_fp32_to_fp64(data_eval_c, data_res_c, m * n, max_error, mean_error);
  cudaSafeCall(cudaFree(data_res_c));

  float duration = 0;
  cudaEvent_t st, ed;
  cudaEventCreate(&st);
  cudaEventCreate(&ed);
  for (int i = 0; i < warmups; i++) {
    ape::apeGemmFP32(apeHandle, ape::APE_TRANS_T, ape::APE_TRANS_N, m, n, k, &alpha_eval, data_eval_a, k, data_eval_b, k, &beta_eval,
                     data_eval_c, m, algo);
  }
  cudaSafeCall(cudaMallocManaged((void **)&data_backup_c, m * n * sizeof(float)));
  cudaSafeCall(cudaMemcpy(data_eval_c, data_backup_c, sizeof(float) * m * n, cudaMemcpyDeviceToDevice));
  for (int i = 0; i < iterations; i++) {
    float tmpdur;
    cudaSafeCall(cudaMemcpy(data_eval_c, data_backup_c, sizeof(float) * m * n, cudaMemcpyDeviceToDevice));
    cudaSafeCall(cudaDeviceSynchronize());
    cudaEventRecord(st, 0);
    ape::apeGemmFP32(apeHandle, ape::APE_TRANS_T, ape::APE_TRANS_N, m, n, k, &alpha_eval, data_eval_a, k, data_eval_b, k, &beta_eval,
                     data_eval_c, m, algo);
    cudaEventRecord(ed, 0);
    cudaEventSynchronize(st);
    cudaEventSynchronize(ed);
    cudaEventElapsedTime(&tmpdur, st, ed);
    duration += tmpdur;
  }
  double perf = double(m) * double(n) * double(k) * 2.0f * iterations / duration / 1e9;

  std::cout << m << "," << n << "," << k << "," << getApeAlgoName(algo)
            << std::fixed << std::setprecision(4) << std::scientific
            << ",,," << mean_error << ","
            << std::fixed << (duration / iterations) << "," << perf << "\n";
  cudaFree(data_eval_a);
  cudaFree(data_eval_b);
  cudaFree(data_eval_c);
  cudaFree(data_backup_c);
  cudaFree(data_res_a);
  cudaFree(data_res_b);
  cudaFree(data_res_c);
}

} // namespace ape
