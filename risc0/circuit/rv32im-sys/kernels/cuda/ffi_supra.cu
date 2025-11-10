// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "eval_check.cuh"

#include "cuda.h"
#include "supra/fp.h"

#include <exception>

namespace risc0::circuit::rv32im_v2::cuda {

__constant__ FpExt poly_mix[kNumPolyMixPows];

namespace {

__device__ __forceinline__ Fp fp_pow_u32(Fp base, uint32_t exp) {
  Fp acc(1);
  Fp cur = base;
  uint32_t e = exp;
  while (e) {
    if (e & 1) {
      acc *= cur;
    }
    e >>= 1;
    if (e) {
      cur *= cur;
    }
  }
  return acc;
}

__device__ __forceinline__ Fp fp_pow_two_pow(Fp base, uint32_t po2) {
  Fp acc = base;
  for (uint32_t i = 0; i < po2; ++i) {
    acc *= acc;
  }
  return acc;
}

} // namespace

// Tight __launch_bounds__ to minimize register pressure - critical for high-register kernels
__global__ __launch_bounds__(256, 2) void eval_check(Fp* __restrict__ check,
                                                     const Fp* __restrict__ ctrl,
                                                     const Fp* __restrict__ data,
                                                     const Fp* __restrict__ accum,
                                                     const Fp* __restrict__ mix,
                                                     const Fp* __restrict__ out,
                                                     const Fp rou,
                                                     uint32_t po2,
                                                     uint32_t domain) {
  const uint32_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t stride = blockDim.x * gridDim.x;
  if (thread_idx >= domain) {
    return;
  }

  const Fp three(3);
  const Fp one(1);
  Fp rou_pow = fp_pow_u32(rou, thread_idx);
  const Fp rou_stride = fp_pow_u32(rou, stride);

  for (uint32_t cycle = thread_idx; cycle < domain; cycle += stride) {
    FpExt tot = poly_fp(cycle, domain, ctrl, out, data, mix, accum);
    Fp base = three * rou_pow;
    Fp y = fp_pow_two_pow(base, po2);
    FpExt ret = tot * inv(y - one);
    check[domain * 0 + cycle] = ret[0];
    check[domain * 1 + cycle] = ret[1];
    check[domain * 2 + cycle] = ret[2];
    check[domain * 3 + cycle] = ret[3];
    rou_pow *= rou_stride;
  }
}

} // namespace risc0::circuit::rv32im_v2::cuda

using namespace risc0::circuit::rv32im_v2::cuda;

extern "C" {

const char* risc0_circuit_rv32im_cuda_eval_check(Fp* check,
                                                 const Fp* ctrl,
                                                 const Fp* data,
                                                 const Fp* accum,
                                                 const Fp* mix,
                                                 const Fp* out,
                                                 const Fp& rou,
                                                 uint32_t po2,
                                                 uint32_t domain,
                                                 const FpExt* poly_mix_pows) {
  try {
    CudaStream stream;
    auto cfg = getSimpleConfig(domain);
    CUDA_OK(cudaMemcpyToSymbol(poly_mix, poly_mix_pows, sizeof(poly_mix)));
    eval_check<<<cfg.grid, cfg.block, 0, stream>>>(
        check, ctrl, data, accum, mix, out, rou, po2, domain);
    CUDA_OK(cudaStreamSynchronize(stream));
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

} // extern "C"
