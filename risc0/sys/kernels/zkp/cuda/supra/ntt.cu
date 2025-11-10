#include "ff/baby_bear.hpp"
#include "ntt/ntt.cuh"

extern "C" RustError::by_value sppark_init() {
  uint32_t lg_domain_size = 1;
  uint32_t domain_size = 1U << lg_domain_size;

  std::vector<fr_t> inout{domain_size};
  inout[0] = fr_t(1);
  inout[1] = fr_t(1);

  const gpu_t& gpu = select_gpu();

  try {
    // Removed cudaDeviceSynchronize() - no need to sync before launch unless debugging
    // This allows overlapping with upstream work

    NTT::Base(gpu,
              &inout[0],
              lg_domain_size,
              NTT::InputOutputOrder::NR,
              NTT::Direction::forward,
              NTT::Type::standard);
    // Removed gpu.sync() - let caller decide when to wait
    // This enables overlapping with subsequent operations
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value sppark_batch_expand(
    fr_t* d_out, fr_t* d_in, uint32_t lg_domain_size, uint32_t lg_blowup, uint32_t poly_count) {
  if (lg_domain_size == 0)
    return RustError{cudaSuccess};

  uint32_t domain_size = 1U << lg_domain_size;
  uint32_t ext_domain_size = domain_size << lg_blowup;

  const gpu_t& gpu = select_gpu();

  try {
    // Removed cudaDeviceSynchronize() - no need to sync before launch unless debugging
    // This allows overlapping with upstream work

    // Process all polynomials - kernels are queued asynchronously
    // The GPU scheduler will keep SMs busy as work becomes available
    for (size_t c = 0; c < poly_count; c++) {
      NTT::LDE_expand(
          gpu, &d_out[c * ext_domain_size], &d_in[c * domain_size], lg_domain_size, lg_blowup);
    }

    // Removed gpu.sync() - let caller decide when to wait
    // This enables overlapping with subsequent operations or PCIe copies
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_batch_NTT(fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count) {
  if (lg_domain_size == 0)
    return RustError{cudaSuccess};

  uint32_t domain_size = 1U << lg_domain_size;

  const gpu_t& gpu = select_gpu();

  try {
    // Removed cudaDeviceSynchronize() - no need to sync before launch unless debugging
    // This allows overlapping with upstream work

    // Process all polynomials - kernels are queued asynchronously
    // The GPU scheduler will keep SMs busy as work becomes available
    for (size_t c = 0; c < poly_count; c++) {
      NTT::Base_dev_ptr(gpu,
                        &d_inout[c * domain_size],
                        lg_domain_size,
                        NTT::InputOutputOrder::RN,
                        NTT::Direction::forward,
                        NTT::Type::standard);
    }

    // Removed gpu.sync() - let caller decide when to wait
    // This enables overlapping with subsequent operations or PCIe copies
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_batch_iNTT(fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count) {
  if (lg_domain_size == 0)
    return RustError{cudaSuccess};

  uint32_t domain_size = 1U << lg_domain_size;

  const gpu_t& gpu = select_gpu();

  try {
    // Removed cudaDeviceSynchronize() - no need to sync before launch unless debugging
    // This allows overlapping with upstream work

    // Process all polynomials - kernels are queued asynchronously
    // The GPU scheduler will keep SMs busy as work becomes available
    for (size_t c = 0; c < poly_count; c++) {
      NTT::Base_dev_ptr(gpu,
                        &d_inout[c * domain_size],
                        lg_domain_size,
                        NTT::InputOutputOrder::NR,
                        NTT::Direction::inverse,
                        NTT::Type::standard);
    }

    // Removed gpu.sync() - let caller decide when to wait
    // This enables overlapping with subsequent operations or PCIe copies
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }

  return RustError{cudaSuccess};
}

extern "C" RustError::by_value
sppark_batch_zk_shift(fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count) {
  if (lg_domain_size == 0)
    return RustError{cudaSuccess};

  uint32_t domain_size = 1U << lg_domain_size;

  const gpu_t& gpu = select_gpu();

  try {
    // Removed cudaDeviceSynchronize() - no need to sync before launch unless debugging
    // This allows overlapping with upstream work

    // Process all polynomials - kernels are queued asynchronously
    // The GPU scheduler will keep SMs busy as work becomes available
    for (size_t c = 0; c < poly_count; c++) {
      NTT::LDE_powers(gpu, &d_inout[c * domain_size], lg_domain_size);
    }

    // Removed gpu.sync() - let caller decide when to wait
    // This enables overlapping with subsequent operations or PCIe copies
  } catch (const cuda_error& e) {
    gpu.sync();
    return RustError{e.code(), e.what()};
  } catch (...) {
    return RustError(cudaErrorUnknown, "Generic exception");
  }

  return RustError{cudaSuccess};
}
