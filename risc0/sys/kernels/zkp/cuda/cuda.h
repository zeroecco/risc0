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

#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>

template <typename... Types> inline std::string fmt(const char* fmt, Types... args) {
  size_t len = std::snprintf(nullptr, 0, fmt, args...);
  std::string ret(++len, '\0');
  std::snprintf(&ret.front(), len, fmt, args...);
  ret.resize(--len);
  return ret;
}

#define CUDA_OK(expr)                                                                              \
  do {                                                                                             \
    cudaError_t code = expr;                                                                       \
    if (code != cudaSuccess) {                                                                     \
      auto file = std::strstr(__FILE__, "sppark");                                                 \
      auto msg = fmt("%s@%s:%d failed: \"%s\"",                                                    \
                     #expr,                                                                        \
                     file ? file : __FILE__,                                                       \
                     __LINE__,                                                                     \
                     cudaGetErrorString(code));                                                    \
      throw std::runtime_error{msg};                                                               \
    }                                                                                              \
  } while (0)

class CudaStream {
private:
  struct Holder {
    cudaStream_t stream;

    Holder() { CUDA_OK(cudaStreamCreate(&stream)); }
    ~Holder() { cudaStreamDestroy(stream); }
  };

public:
  CudaStream() = default;
  ~CudaStream() = default;

  static cudaStream_t get() {
    thread_local Holder holder;
    return holder.stream;
  }

  inline operator cudaStream_t() const { return get(); }
};

struct LaunchConfig {
  dim3 grid;
  dim3 block;
  size_t shared;

  LaunchConfig(dim3 grid, dim3 block, size_t shared = 0)
      : grid(grid), block(block), shared(shared) {}
  LaunchConfig(int grid, int block, size_t shared = 0) : grid(grid), block(block), shared(shared) {}
};

inline LaunchConfig getSimpleConfig(uint32_t count) {
  int device;
  CUDA_OK(cudaGetDevice(&device));

  int maxThreads;
  CUDA_OK(cudaDeviceGetAttribute(&maxThreads, cudaDevAttrMaxThreadsPerBlock, device));

  int block = maxThreads / 4;
  int grid = (count + block - 1) / block;
  return LaunchConfig{grid, block, 0};
}

inline int getEnvBlockSize(const char* name, int defaultBlock) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return defaultBlock;
  }

  char* end = nullptr;
  long parsed = std::strtol(raw, &end, 10);
  if (end == raw || (end != nullptr && *end != '\0') || parsed <= 0) {
    return defaultBlock;
  }

  return static_cast<int>(parsed);
}

inline LaunchConfig getBlockConfig(uint32_t count, int block) {
  int device;
  CUDA_OK(cudaGetDevice(&device));

  int maxThreads;
  CUDA_OK(cudaDeviceGetAttribute(&maxThreads, cudaDevAttrMaxThreadsPerBlock, device));

  if (block <= 0) {
    block = maxThreads / 4;
  }
  block = std::min(block, maxThreads);

  int grid = (count + block - 1) / block;
  return LaunchConfig{grid, block, 0};
}

inline int roundDownPowerOfTwo(int value) {
  int out = 1;
  while (out <= value / 2) {
    out *= 2;
  }
  return out;
}

template <typename... ExpTypes, typename... ActTypes>
const char* launchKernel(void (*kernel)(ExpTypes...),
                         uint32_t count,
                         uint32_t shared_size,
                         ActTypes&&... args) {
  try {
    CudaStream stream;
    LaunchConfig cfg = getSimpleConfig(count);
    kernel<<<cfg.grid, cfg.block, shared_size, stream>>>(std::forward<ActTypes>(args)...);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaStreamSynchronize(stream));
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}

template <typename... ExpTypes, typename... ActTypes>
const char* launchKernelWithBlock(void (*kernel)(ExpTypes...),
                                  uint32_t count,
                                  uint32_t block,
                                  uint32_t shared_size,
                                  ActTypes&&... args) {
  try {
    CudaStream stream;
    LaunchConfig cfg = getBlockConfig(count, block);
    kernel<<<cfg.grid, cfg.block, shared_size, stream>>>(std::forward<ActTypes>(args)...);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaStreamSynchronize(stream));
  } catch (const std::exception& err) {
    return strdup(err.what());
  } catch (...) {
    return strdup("Generic exception");
  }
  return nullptr;
}
