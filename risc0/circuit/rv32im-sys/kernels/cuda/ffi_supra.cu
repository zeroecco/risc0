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

#ifndef EVAL_CHECK_MAX_THREADS
#define EVAL_CHECK_MAX_THREADS 256
#endif

#ifndef EVAL_CHECK_MIN_BLOCKS
#define EVAL_CHECK_MIN_BLOCKS 1
#endif

#ifndef EVAL_CHECK_COMBINED_TU
__constant__ FpExt poly_mix[kNumPolyMixPows];
#endif

__global__ __launch_bounds__(EVAL_CHECK_MAX_THREADS, EVAL_CHECK_MIN_BLOCKS) void eval_check(
    Fp* __restrict__ check,
    const Fp* __restrict__ ctrl,
    const Fp* __restrict__ data,
    const Fp* __restrict__ accum,
    const Fp* __restrict__ mix,
    const Fp* __restrict__ out,
    const Fp rou,
    uint32_t po2,
    uint32_t domain) {
  asm volatile(".pragma \"enable_smem_spilling\";\n");

  uint32_t cycle = blockDim.x * blockIdx.x + threadIdx.x;
  if (cycle < domain) {
    FpExt tot = poly_fp(cycle, domain, ctrl, out, data, mix, accum);
    Fp x = pow(rou, cycle);
    Fp y = pow(Fp(3) * x, 1 << po2);
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
                                                 const FpExt* poly_mix_pows) {
  try {
    CudaStream stream;
    auto cfg = getBlockConfig(
        domain, getEnvBlockSize("RISC0_EVAL_CHECK_BLOCK_SIZE", EVAL_CHECK_MAX_THREADS));
    CUDA_OK(cudaMemcpyToSymbolAsync(
        poly_mix, poly_mix_pows, sizeof(poly_mix), 0, cudaMemcpyHostToDevice, stream));
    eval_check<<<cfg.grid, cfg.block, 0, stream>>>(
        check, ctrl, data, accum, mix, out, rou, po2, domain);
    CUDA_OK(cudaGetLastError());
    if (!CudaStream::use_chained_default_stream()) {
      CUDA_OK(cudaStreamSynchronize(stream));
    }
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

} // extern "C"
