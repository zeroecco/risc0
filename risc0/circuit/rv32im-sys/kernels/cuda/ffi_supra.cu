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
__constant__ Fp kPolyFpLut[kNumFpConstants];

// Precomputed rou powers table - avoids expensive pow() calls per thread
// Host should populate this before kernel launch
__constant__ Fp rou_pows[1]; // Will be sized to domain size at runtime
// Note: For large domains, consider using device memory instead of constant memory
// due to 64KB constant memory limit

__global__ void eval_check(Fp* check,
                           const Fp* __restrict__ ctrl,
                           const Fp* __restrict__ data,
                           const Fp* __restrict__ accum,
                           const Fp* __restrict__ mix,
                           const Fp* __restrict__ out,
                           const Fp* __restrict__ rou_pows_table, // Precomputed rou^cycle
                           const Fp* __restrict__ y_pows_table,   // Precomputed (3*rou^cycle)^(1<<po2)
                           uint32_t domain) {
  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle < domain) {
    FpExt tot = poly_fp(cycle, domain, ctrl, out, data, mix, accum);
    // Use precomputed powers instead of expensive pow() calls
    // pow(rou, cycle) -> rou_pows_table[cycle]
    // pow(Fp(3) * x, 1 << po2) -> y_pows_table[cycle]
    Fp y = y_pows_table[cycle];
    FpExt ret = tot * inv(y - Fp(1));
    check[domain * 0 + cycle] = ret[0];
    check[domain * 1 + cycle] = ret[1];
    check[domain * 2 + cycle] = ret[2];
    check[domain * 3 + cycle] = ret[3];
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
                                                 const FpExt* poly_mix_pows,
                                                 const Fp* rou_pows_table,  // Precomputed rou^cycle
                                                 const Fp* y_pows_table) {   // Precomputed (3*rou^cycle)^(1<<po2)
  try {
    // Removed cudaDeviceSynchronize() - no need to sync before launch unless debugging
    // This allows overlapping with upstream work (witgen, accum)

    CudaStream stream;
    auto cfg = getSimpleConfig(domain);

    // Copy constant tables
    CUDA_OK(cudaMemcpyToSymbol(poly_mix, poly_mix_pows, sizeof(poly_mix)));

    // Initialize Fp constants LUT (values should match eval_check_0.cu usage)
    // TODO: Generate this table automatically from the code generator
    Fp fp_constants[kNumFpConstants] = {
      Fp(51), Fp(1073725472), Fp(1073725440), Fp(32768), Fp(8192),
      Fp(2048), Fp(512), Fp(128), Fp(32), Fp(16),
      Fp(4096), Fp(1024), Fp(256), Fp(64), Fp(61440),
      Fp(2013265920), Fp(65535), Fp(49151), Fp(16384), Fp(48),
      Fp(8), Fp(9), Fp(10), Fp(11), Fp(12),
      Fp(2), Fp(3), Fp(4), Fp(5), Fp(6),
      Fp(7), Fp(1), Fp(0)
    };
    CUDA_OK(cudaMemcpyToSymbol(kPolyFpLut, fp_constants, sizeof(fp_constants)));

    eval_check<<<cfg.grid, cfg.block, 0, stream>>>(
        check, ctrl, data, accum, mix, out, rou_pows_table, y_pows_table, domain);

    // Removed cudaStreamSynchronize() - let caller decide when to wait
    // This enables overlapping PCIe copies or subsequent kernels
    // Caller can use cudaStreamSynchronize(stream) if they need to wait
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

} // extern "C"
