#include <ff/alt_bn128.hpp>
#include <ff/baby_bear.hpp>
#include <util/gpu_t.cuh>
#include <util/rusterror.h>

#include <polynomial/div_by_x_minus_z.cuh>
#include <polynomial/prefix_op.cuh>

#include <cstdlib>

#include "poseidon2.cuh"
#include "poseidon254.cuh"

static size_t get_env_block_size(const char* name, size_t default_block) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return default_block;
  }

  char* end = nullptr;
  long parsed = std::strtol(raw, &end, 10);
  if (end == raw || (end != nullptr && *end != '\0') || parsed <= 0) {
    return default_block;
  }

  return static_cast<size_t>(parsed);
}

extern "C" RustError::by_value
sppark_poseidon2_fold(poseidon_out_t* d_out, const poseidon_in_t* d_in, size_t num_hashes) {
  const gpu_t& gpu = select_gpu();

  size_t block_size = std::min<size_t>(get_env_block_size("RISC0_POSEIDON2_FOLD_BLOCK_SIZE", 128), 256);
  block_size = std::min(block_size, num_hashes);
  block_size = std::max<size_t>(block_size, 1);
  size_t num_blocks = (num_hashes + block_size - 1) / block_size;

  try {
    _poseidon2_fold<<<num_blocks, block_size, 0, gpu>>>(d_out, d_in, num_hashes);

    CUDA_OK(cudaGetLastError());
    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_poseidon2_rows(poseidon_out_t* d_out, const fr_t* d_in, uint32_t count, uint32_t col_size) {
  const gpu_t& gpu = select_gpu();

  size_t block_size = std::min<size_t>(get_env_block_size("RISC0_POSEIDON2_ROWS_BLOCK_SIZE", 128), 256);
  block_size = std::min<size_t>(block_size, count);
  block_size = std::max<size_t>(block_size, 1);
  size_t num_blocks = (count + block_size - 1) / block_size;

  try {
    _poseidon2_rows<<<num_blocks, block_size, 0, gpu>>>(d_out, d_in, count, col_size);

    CUDA_OK(cudaGetLastError());
    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value sppark_poseidon2_merkle_tree(
    poseidon_out_t* d_nodes,
    const fr_t* d_matrix,
    uint32_t rows,
    uint32_t col_size,
    uint32_t layers) {
  const gpu_t& gpu = select_gpu();

  size_t row_block_size =
      std::min<size_t>(get_env_block_size("RISC0_POSEIDON2_ROWS_BLOCK_SIZE", 128), 256);
  row_block_size = std::min<size_t>(row_block_size, rows);
  row_block_size = std::max<size_t>(row_block_size, 1);
  size_t row_blocks = (rows + row_block_size - 1) / row_block_size;

  try {
    poseidon_out_t* d_leaves = d_nodes + rows;
    _poseidon2_rows<<<row_blocks, row_block_size, 0, gpu>>>(d_leaves, d_matrix, rows, col_size);
    CUDA_OK(cudaGetLastError());

    for (uint32_t i = layers; i-- > 0;) {
      uint32_t layer_size = 1u << i;
      size_t fold_block_size =
          std::min<size_t>(get_env_block_size("RISC0_POSEIDON2_FOLD_BLOCK_SIZE", 128), 256);
      fold_block_size = std::min<size_t>(fold_block_size, layer_size);
      fold_block_size = std::max<size_t>(fold_block_size, 1);
      size_t fold_blocks = (layer_size + fold_block_size - 1) / fold_block_size;
      poseidon_out_t* d_output = d_nodes + layer_size;
      const poseidon_in_t* d_input = reinterpret_cast<const poseidon_in_t*>(d_nodes + layer_size * 2);
      _poseidon2_fold<<<fold_blocks, fold_block_size, 0, gpu>>>(d_output, d_input, layer_size);
      CUDA_OK(cudaGetLastError());
    }

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

static void compute_grid_block_size(size_t total_count, size_t& block_size, size_t& num_blocks) {
  size_t min_block_size = 4 * WARP_SZ;

  if (total_count < (block_size * num_blocks)) {
    size_t count_per_block = total_count / num_blocks;

    if (count_per_block > min_block_size) {
      block_size = ((count_per_block + min_block_size - 1) / min_block_size) * min_block_size;
      num_blocks = (total_count + block_size - 1) / block_size;
    } else {
      block_size = min_block_size;
      num_blocks = (total_count + min_block_size - 1) / min_block_size;
    }
  } else {
    size_t base_iter = (total_count + (num_blocks * block_size) - 1) / (num_blocks * block_size);
    size_t out_block_size = block_size;

    for (size_t cur_block_size = block_size - min_block_size; cur_block_size >= min_block_size;
         cur_block_size -= min_block_size) {
      size_t cur_iter =
          (total_count + (num_blocks * cur_block_size) - 1) / (num_blocks * cur_block_size);

      if (cur_iter != base_iter)
        break;
      out_block_size = cur_block_size;
    }

    block_size = out_block_size;
  }
}

extern "C" RustError::by_value
sppark_poseidon254_fold(alt_bn128::fr_t* d_out, const alt_bn128::fr_t* d_in, size_t num_hashes) {
  const gpu_t& gpu = select_gpu();

  size_t block_size = 512;
  size_t num_blocks = gpu.sm_count();

  compute_grid_block_size(num_hashes, block_size, num_blocks);

  try {
    _poseidon254_fold<<<num_blocks, block_size, 0, gpu>>>(d_out, d_in, num_hashes);

    CUDA_OK(cudaGetLastError());

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_poseidon254_rows(alt_bn128::fr_t* d_out, const fr_t* d_in, size_t count, uint32_t col_size) {
  const gpu_t& gpu = select_gpu();

  size_t block_size = 512;
  size_t num_blocks = gpu.sm_count();

  compute_grid_block_size(count, block_size, num_blocks);

  try {
    _poseidon254_rows<<<num_blocks, block_size, 0, gpu>>>(d_out, d_in, count, col_size);

    CUDA_OK(cudaGetLastError());

    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value sppark_prefix_product(fr4_t d_inout[/*count*/], uint32_t count) {
  const gpu_t& gpu = select_gpu();

  try {
    prefix_op<Multiply<fr4_t>>(d_inout, count, gpu);
    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
supra_poly_divide(fr4_t d_inout[/*len*/], size_t len, fr4_t* remainder, const fr4_t& pow) {
  const gpu_t& gpu = select_gpu();

  try {
    div_by_x_minus_z<true>(d_inout, len, pow, gpu);
    gpu.DtoH(remainder, &d_inout[len - 1], 1);
    gpu.sync();
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  }

  return RustError{cudaSuccess};
}
