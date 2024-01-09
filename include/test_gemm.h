#pragma once

#include "ape.h"

namespace ape {
void test_gemm_fp32(int m, int n, int k, ape::ApeAlgo algo, int warmups = 2, int iterations = 10);
void test_gemm_fp32(size_t seed, float alpha, float beta, float mean, float stdev,
                    int m, int n, int k, ape::ApeAlgo algo, int warmups = 2, int iterations = 10);
} // namespace ape
