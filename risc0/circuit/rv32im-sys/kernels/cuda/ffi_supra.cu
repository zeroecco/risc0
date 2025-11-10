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

// OPTIMIZATION: Two-phase kernel approach for better GPU utilization
// Phase 1: Compute poly_fp with lower register pressure (process fewer cycles per thread)
// Phase 2: Finalize with y computation (very low register pressure, can overlap)

// Phase 1: Compute poly_fp results and store in intermediate buffer
// This phase can run with better occupancy by processing fewer cycles per thread
// Relaxed __launch_bounds__ to allow rv32im_v2_19's 255 registers
__global__ __launch_bounds__(256, 1) void eval_check_phase1(FpExt* __restrict__ poly_results,
                                                             const Fp* __restrict__ ctrl,
                                                             const Fp* __restrict__ data,
                                                             const Fp* __restrict__ accum,
                                                             const Fp* __restrict__ mix,
                                                             const Fp* __restrict__ out,
                                                             uint32_t domain) {
  const uint32_t thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (thread_idx >= domain) {
    return;
  }

  // Process 1 cycle per thread for maximum occupancy
  // This reduces register pressure by not keeping multiple cycles in registers
  FpExt tot = poly_fp(thread_idx, domain, ctrl, out, data, mix, accum);
  poly_results[thread_idx] = tot;
}

// Phase 2: Finalize computation with y values (very low register pressure)
// This can run with maximum occupancy and overlap with phase 1
__global__ __launch_bounds__(256, 4) void eval_check_phase2(Fp* __restrict__ check,
                                                              const FpExt* __restrict__ poly_results,
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

  // OPTIMIZATION: Precompute y_stride to avoid expensive fp_pow_two_pow in loop
  const Fp y_stride = fp_pow_two_pow(three * rou_stride, po2);
  Fp base = three * rou_pow;
  Fp y = fp_pow_two_pow(base, po2);

  for (uint32_t cycle = thread_idx; cycle < domain; cycle += stride) {
    // Load intermediate poly_fp result
    FpExt tot = poly_results[cycle];

    FpExt ret = tot * inv(y - one);
    check[domain * 0 + cycle] = ret[0];
    check[domain * 1 + cycle] = ret[1];
    check[domain * 2 + cycle] = ret[2];
    check[domain * 3 + cycle] = ret[3];
    y *= y_stride;
  }
}

// Original single-phase kernel (kept for comparison/fallback)
__global__ __launch_bounds__(256, 1) void eval_check_original(Fp* __restrict__ check,
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

  // OPTIMIZATION: Precompute y_stride to avoid expensive fp_pow_two_pow in loop
  const Fp y_stride = fp_pow_two_pow(three * rou_stride, po2);
  Fp base = three * rou_pow;
  Fp y = fp_pow_two_pow(base, po2);

  for (uint32_t cycle = thread_idx; cycle < domain; cycle += stride) {
    FpExt tot = poly_fp(cycle, domain, ctrl, out, data, mix, accum);
    FpExt ret = tot * inv(y - one);
    check[domain * 0 + cycle] = ret[0];
    check[domain * 1 + cycle] = ret[1];
    check[domain * 2 + cycle] = ret[2];
    check[domain * 3 + cycle] = ret[3];
    y *= y_stride;
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
    CUDA_OK(cudaMemcpyToSymbol(poly_mix, poly_mix_pows, sizeof(poly_mix)));

    // OPTIMIZATION: Two-phase approach for better GPU utilization
    // Benefits:
    // 1. Phase 1: Process 1 cycle per thread (no y values in registers across cycles)
    // 2. Phase 2: Very low register pressure (occupancy=4 vs occupancy=1 in original)
    // 3. Phase 2 can potentially overlap with Phase 1 using multiple streams
    // Tradeoff: Extra memory allocation and write/read (domain * sizeof(FpExt))
    //
    // Note: Phase 1 still has high register pressure (255 registers from poly_fp),
    // but Phase 2's better occupancy (4 blocks/SM) can improve overall throughput.
    // For very large domains, consider using multiple streams to overlap phases.

    // Allocate intermediate buffer for poly_fp results
    FpExt* d_poly_results = nullptr;
    CUDA_OK(cudaMalloc(&d_poly_results, domain * sizeof(FpExt)));

    // Phase 1: Compute poly_fp for all cycles
    // Process 1 cycle per thread to avoid keeping y values in registers
    auto cfg1 = getSimpleConfig(domain);
    eval_check_phase1<<<cfg1.grid, cfg1.block, 0, stream>>>(
        d_poly_results, ctrl, data, accum, mix, out, domain);

    // Phase 2: Finalize computation with low register pressure
    // This phase can run with occupancy=4 (256, 4) for much better GPU utilization
    auto cfg2 = getSimpleConfig(domain);
    eval_check_phase2<<<cfg2.grid, cfg2.block, 0, stream>>>(
        check, d_poly_results, rou, po2, domain);

    CUDA_OK(cudaStreamSynchronize(stream));
    CUDA_OK(cudaFree(d_poly_results));
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

} // extern "C"
