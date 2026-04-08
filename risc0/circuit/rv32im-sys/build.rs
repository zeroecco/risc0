// Copyright 2025 RISC Zero, Inc.
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

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

use risc0_build_kernel::{KernelBuild, KernelType};

fn main() {
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        build_cuda_kernels();
    }

    build_cpu_kernels();
}

fn build_cpu_kernels() {
    rerun_if_changed("kernels/cxx");
    KernelBuild::new(KernelType::Cpp)
        .files(glob_paths("kernels/cxx/*.cpp"))
        .deps(glob_paths("kernels/cxx/*.h"))
        .deps(glob_paths("kernels/cxx/*.cpp.inc"))
        .deps(glob_paths("kernels/cxx/*.h.inc"))
        .include(env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap())
        .compile("risc0_rv32im_cpu");
}

fn build_cuda_kernels() {
    let output = "risc0_rv32im_cuda";
    let eval_output = "risc0_rv32im_cuda_eval";

    println!("cargo:rerun-if-env-changed=RISC0_EVAL_CHECK_OBJECT");
    println!("cargo:rerun-if-env-changed=RISC0_EVAL_CHECK_MAX_THREADS");
    println!("cargo:rerun-if-env-changed=RISC0_EVAL_CHECK_MIN_BLOCKS");
    println!("cargo:rerun-if-env-changed=RISC0_EVAL_CHECK_MAXRREGCOUNT");
    println!("cargo:rerun-if-env-changed=NVCC_APPEND_FLAGS");
    println!("cargo:rerun-if-env-changed=NVCC_PREPEND_FLAGS");
    println!("cargo:rerun-if-env-changed=SCCACHE_RECACHE");
    rerun_if_changed("kernels/cuda");

    env::set_var("SCCACHE_IDLE_TIMEOUT", "0");

    if env::var("RISC0_SKIP_BUILD_KERNELS").is_ok() {
        let out_dir = env::var("OUT_DIR").map(PathBuf::from).unwrap();
        let out_path = out_dir.join(format!("lib{output}-skip.a"));
        std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&out_path)
            .unwrap();
        println!("cargo:{}={}", output, out_path.display());
        return;
    }

    build_eval_check_kernel(eval_output);

    let mut build = cc::Build::new();
    build
        .cuda(true)
        .cudart("static")
        .debug(false)
        .flag("-diag-suppress=177")
        .flag("-diag-suppress=550")
        .flag("-diag-suppress=2922")
        .flag("-std=c++17")
        .flag("-Xcompiler")
        .flag("-Wno-unused-function,-Wno-unused-parameter")
        .flag("-Xcompiler")
        .flag("-O3")
        .flag("-Xptxas")
        .flag("-O3")
        .include(env::var("DEP_RISC0_SYS_CUDA_ROOT").unwrap())
        .include(env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap())
        .include(env::var("DEP_SPPARK_ROOT").unwrap());
    if env::var_os("NVCC_PREPEND_FLAGS").is_none() && env::var_os("NVCC_APPEND_FLAGS").is_none() {
        build.flag("-arch=native");
    }

    let files = glob_paths("kernels/cuda/*.cu")
        .into_iter()
        .filter(|path| {
            let name = path.file_name().and_then(|x| x.to_str()).unwrap_or_default();
            !matches!(
                name,
                "eval_check_0.cu"
                    | "eval_check_1.cu"
                    | "eval_check_2.cu"
                    | "eval_check_3.cu"
                    | "ffi_supra.cu"
                    | "eval_check_combined.cu"
                    | "steps.cu"
            )
        })
        .collect::<Vec<_>>();
    build.files(files).file("kernels/cuda/steps.cu").compile(output);
}

fn build_eval_check_kernel(output: &str) {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let obj_path = out_dir.join("eval_check_combined.o");
    let lib_path = out_dir.join(format!("lib{output}.a"));
    let cuda_root = env::var("DEP_RISC0_SYS_CUDA_ROOT").unwrap();
    let cxx_root = env::var("DEP_RISC0_SYS_CXX_ROOT").unwrap();
    let sppark_root = env::var("DEP_SPPARK_ROOT").unwrap();
    let max_threads = env::var("RISC0_EVAL_CHECK_MAX_THREADS").unwrap_or_else(|_| "256".into());
    let min_blocks = env::var("RISC0_EVAL_CHECK_MIN_BLOCKS").unwrap_or_else(|_| "1".into());

    if let Ok(reuse_obj) = env::var("RISC0_EVAL_CHECK_OBJECT") {
        let reuse_path = PathBuf::from(&reuse_obj);
        if reuse_path != obj_path {
            std::fs::copy(&reuse_obj, &obj_path).unwrap_or_else(|e| {
                panic!(
                    "Failed to copy prebuilt eval_check object {reuse_obj} -> {}: {e}",
                    obj_path.display()
                )
            });
        }
        build_eval_check_archive(output, &obj_path, &lib_path);
        return;
    }

    let mut cmd = Command::new("nvcc");
    cmd.arg("-c")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-Xptxas")
        .arg("-O3,-v")
        .arg("-Xcompiler")
        .arg("-O3,-Wno-unused-function,-Wno-unused-parameter")
        .arg("-diag-suppress=177")
        .arg("-diag-suppress=550")
        .arg("-diag-suppress=2922")
        .arg(format!("-DEVAL_CHECK_MAX_THREADS={max_threads}"))
        .arg(format!("-DEVAL_CHECK_MIN_BLOCKS={min_blocks}"))
        .arg(format!("-I{cuda_root}"))
        .arg(format!("-I{cxx_root}"))
        .arg(format!("-I{sppark_root}"));

    if let Ok(maxrregcount) = env::var("RISC0_EVAL_CHECK_MAXRREGCOUNT") {
        cmd.arg(format!("-maxrregcount={maxrregcount}"));
    }

    if env::var_os("NVCC_PREPEND_FLAGS").is_none() && env::var_os("NVCC_APPEND_FLAGS").is_none() {
        cmd.arg("-arch=native");
    }

    cmd.arg("-o")
        .arg(&obj_path)
        .arg("kernels/cuda/eval_check_combined.cu");

    let output_res = cmd.output().unwrap_or_else(|e| {
        panic!("Failed to invoke nvcc for eval_check_combined.cu: {e}");
    });

    let stdout_str = String::from_utf8_lossy(&output_res.stdout);
    let stderr_str = String::from_utf8_lossy(&output_res.stderr);
    for line in stdout_str.lines().chain(stderr_str.lines()) {
        if !line.is_empty() {
            println!("cargo:warning={line}");
        }
    }
    assert!(
        output_res.status.success(),
        "nvcc failed to compile eval_check_combined.cu (exit code: {})",
        output_res.status
    );

    build_eval_check_archive(output, &obj_path, &lib_path);
}

fn build_eval_check_archive(output: &str, obj_path: &Path, lib_path: &Path) {
    let status = Command::new("ar")
        .arg("rcs")
        .arg(lib_path)
        .arg(obj_path)
        .status()
        .expect("Failed to invoke ar for eval_check_combined");
    assert!(status.success(), "ar failed for eval_check_combined");

    let out_dir = lib_path.parent().unwrap();
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static={output}");
}

fn rerun_if_changed<P: AsRef<Path>>(path: P) {
    println!("cargo:rerun-if-changed={}", path.as_ref().display());
}

fn glob_paths(pattern: &str) -> Vec<PathBuf> {
    glob::glob(pattern).unwrap().map(|x| x.unwrap()).collect()
}
