#pragma once

namespace ape {
void test_gemm_fp32(int m, int n, int k, ape::ApeAlgo algo, int iterations = 10);
} // namespace ape
