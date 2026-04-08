// Self-contained eval_check compilation unit.
// Compiled with whole-program CUDA compilation (no RDC / -dc) so that
// ptxas can optimize the current frozen eval_check implementation as one TU.

#define EVAL_CHECK_COMBINED_TU

// ---- eval_check sub-functions (rv32im_v2_0..19) + poly_fp ----
#include "eval_check_0.cu"
#include "eval_check_1.cu"
#include "eval_check_2.cu"
#include "eval_check_3.cu"

// ---- __global__ eval_check kernel + FFI wrapper ----
#include "ffi_supra.cu"
