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

use crate::components::Component;
use crate::env::Environment;
use crate::error::{Result, RzupError};
use crate::events::RzupEvent;

use chrono::{Datelike, Utc};
use semver::Version;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const CONFIG_TOML: &str = include_str!("rust-toolchain-build-config.toml");

fn copy_dir_all(src: &Path, dst: &Path) -> Result<()> {
    if !src.is_dir() {
        return Err(RzupError::Other(format!(
            "source is not a directory: {}",
            src.display()
        )));
    }

    if dst.exists() {
        fs::remove_dir_all(dst)?;
    }
    fs::create_dir_all(dst)?;

    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        if src_path.is_dir() {
            copy_dir_all(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)?;
        }
    }

    Ok(())
}

fn stream_lines(
    reader: impl BufRead,
    mut output_cb: impl FnMut(&str) + Clone + Send + Sync,
) -> Result<String> {
    let mut output = String::new();
    for line in reader.lines() {
        let line = line.map_err(|e| RzupError::Other(e.to_string()))?;
        output_cb(&line);
        output += &line;
        output += "\n";
    }
    Ok(output)
}

pub fn run_command_and_stream_output(
    program: &str,
    args: &[&str],
    current_dir: Option<&Path>,
    env: &[(&str, &str)],
    output_cb: impl FnMut(&str) + Clone + Send + Sync,
) -> Result<String> {
    let mut cmd = Command::new(program);
    cmd.args(args);
    cmd.envs(env.iter().copied());

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    if let Some(path) = current_dir {
        cmd.current_dir(path);
    }
    let mut child = cmd.spawn()?;

    std::thread::scope(|scope| {
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let stderr = BufReader::new(child.stderr.take().unwrap());

        let output_cb_clone = output_cb.clone();
        let stdout_thread = scope.spawn(move || stream_lines(stdout, output_cb_clone));

        let output_cb_clone = output_cb.clone();
        let stderr_thread = scope.spawn(move || stream_lines(stderr, output_cb_clone));

        let status = child.wait()?;

        let stdout_res = stdout_thread.join().unwrap();
        let stderr_res = stderr_thread.join().unwrap();

        if !status.success() {
            let stdout = stdout_res.unwrap_or_default();
            let stderr = stderr_res.unwrap_or_default();
            let cmd_str =
                Vec::from_iter(std::iter::once(program).chain(args.iter().copied())).join(" ");

            return Err(RzupError::Other(format!(
                "Process `{cmd_str}` failed with status {status}\n\
                stodout: {stdout}\n\
                stderr: {stderr}",
            )));
        }
        stdout_res
    })
}

pub fn run_command(
    program: &str,
    args: &[&str],
    current_dir: Option<&Path>,
    env: &[(&str, &str)],
) -> Result<String> {
    run_command_and_stream_output(program, args, current_dir, env, |_| {})
}

fn git_clone(src: &str, dest: &Path) -> Result<()> {
    run_command(
        "git",
        &[
            "clone",
            "--recurse-submodules",
            src,
            dest.to_str()
                .ok_or_else(|| RzupError::Other("non-UTF8 path".into()))?,
        ],
        None,
        &[],
    )?;
    Ok(())
}

fn git_checkout(path: &Path, tag_or_commit: &str) -> Result<()> {
    run_command("git", &["checkout", "-f", tag_or_commit], Some(path), &[])?;
    Ok(())
}

fn git_fetch(path: &Path) -> Result<()> {
    run_command("git", &["fetch", "--all", "--prune"], Some(path), &[])?;
    Ok(())
}

fn git_reset_hard(path: &Path) -> Result<()> {
    run_command("git", &["reset", "--hard"], Some(path), &[])?;
    Ok(())
}

fn git_submodule_update(path: &Path) -> Result<()> {
    run_command(
        "git",
        &["submodule", "update", "--init", "--recursive"],
        Some(path),
        &[],
    )?;
    Ok(())
}

pub fn git_short_rev_parse(path: &Path, tag: &str) -> Result<String> {
    Ok(
        run_command("git", &["rev-parse", "--short", tag], Some(path), &[])?
            .trim()
            .into(),
    )
}

fn find_build_directories(build_dir: &Path) -> Result<(PathBuf, PathBuf, PathBuf)> {
    if !build_dir.exists() {
        return Err(RzupError::Other(
            "failed to find Rust toolchain build directory".into(),
        ));
    }

    for entry in std::fs::read_dir(build_dir)? {
        let entry = entry?;
        let stage2 = entry.path().join("stage2");
        let stage2_tools_bin = entry.path().join("stage2-tools-bin");
        let stage3 = entry.path().join("stage3");
        if stage2.is_dir() && stage2_tools_bin.is_dir() && stage3.is_dir() {
            return Ok((stage2, stage2_tools_bin, stage3));
        }
    }
    Err(RzupError::Other(
        "failed to find Rust toolchain stage2/stage3 build directories".into(),
    ))
}

pub fn build_rust_toolchain(
    env: &Environment,
    repo_url: &str,
    tag_or_commit: &Option<String>,
    path: &Option<String>,
) -> Result<Version> {
    env.emit(RzupEvent::BuildingRustToolchain);

    let _lock_file = env.flock("rust-toolchain-build")?;

    // if building from commit
    let repo_dir = match path {
        None => {
            let repo_dir = env.tmp_dir().join("build-rust-toolchain");
            if !repo_dir.join(".git").exists() {
                env.emit(RzupEvent::BuildingRustToolchainUpdate {
                    message: "cloning git repository".into(),
                });
                git_clone(repo_url, &repo_dir)?;
            } else {
                git_fetch(&repo_dir)?;
            }
            let tag_or_commit = tag_or_commit.as_ref().unwrap();
            git_checkout(&repo_dir, tag_or_commit)?;
            git_reset_hard(&repo_dir)?;
            git_submodule_update(&repo_dir)?;
            repo_dir
        }
        Some(path) => path.into(),
    };

    let commit = git_short_rev_parse(&repo_dir, "HEAD")?;

    let version_str = std::fs::read_to_string(repo_dir.join("src/version")).map_err(|e| {
        RzupError::Other(format!("failed to read Rust version from repository: {e}"))
    })?;
    let mut version = Version::parse(version_str.trim())?;
    version.build = semver::BuildMetadata::new(&commit).unwrap();

    let dest_dir = Component::RustToolchain.get_version_dir(env, &version);
    if dest_dir.exists() {
        return Err(RzupError::Other(format!(
            "Rust toolchain version {version} already installed"
        )));
    }

    std::fs::write(repo_dir.join("config.toml"), CONFIG_TOML)?;

    let req = semver::VersionReq::parse(">=1.82.0")?;
    let lower_atomic = if req.matches(&version) {
        "-Cpasses=lower-atomic"
    } else {
        "-Cpasses=loweratomic"
    };

    for stage in [None, Some(2), Some(3)] {
        let mut args = vec!["build".into()];
        if let Some(stage) = stage {
            args.push("--stage".into());
            args.push(stage.to_string());
        }

        env.emit(RzupEvent::BuildingRustToolchainUpdate {
            message: format!("./x {}", args.join(" ")),
        });

        run_command_and_stream_output(
            "./x",
            &args.iter().map(|a| a.as_ref()).collect::<Vec<_>>(),
            Some(&repo_dir),
            &[(
                "CARGO_TARGET_RISCV32IM_RISC0_ZKVM_ELF_RUSTFLAGS",
                lower_atomic,
            )],
            |line| {
                env.emit(RzupEvent::BuildingRustToolchainUpdate {
                    message: line.into(),
                });
            },
        )
        .map_err(|e| {
            RzupError::Other(format!(
                "failed to run Rust toolchain build stage={stage:?}: {e}"
            ))
        })?;
    }

    env.emit(RzupEvent::BuildingRustToolchainUpdate {
        message: "installing".into(),
    });

    if let Some(parent) = dest_dir.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let (stage2, stage2_tools, stage3) = find_build_directories(&repo_dir.join("build"))?;
    std::fs::rename(stage2, &dest_dir)?;

    let riscv_libs = "lib/rustlib/riscv32im-risc0-zkvm-elf";

    let riscv_libs_dest = dest_dir.join(riscv_libs);
    if let Some(parent) = riscv_libs_dest.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::rename(stage3.join(riscv_libs), riscv_libs_dest)?;

    for tool in std::fs::read_dir(stage2_tools)? {
        let tool = tool?;
        std::fs::rename(tool.path(), dest_dir.join("bin").join(tool.file_name()))?;
    }

    env.emit(RzupEvent::DoneBuildingRustToolchain {
        version: version.to_string(),
    });

    Ok(version)
}

pub fn build_cpp_toolchain(
    env: &Environment,
    repo_url: &str,
    tag_or_commit: &Option<String>,
    path: &Option<String>,
) -> Result<Version> {
    env.emit(RzupEvent::BuildingCppToolchain);

    let _lock_file = env.flock("cpp-toolchain-build")?;

    // if building from commit
    let repo_dir = match path {
        None => {
            let repo_dir = env.tmp_dir().join("build-cpp-toolchain");
            if !repo_dir.join(".git").exists() {
                env.emit(RzupEvent::BuildingCppToolchainUpdate {
                    message: "cloning git repository".into(),
                });
                git_clone(repo_url, &repo_dir)?;
            } else {
                git_fetch(&repo_dir)?;
            }
            let tag_or_commit = tag_or_commit.as_ref().unwrap();
            git_checkout(&repo_dir, tag_or_commit)?;
            git_reset_hard(&repo_dir)?;
            git_submodule_update(&repo_dir)?;
            repo_dir
        }
        Some(path) => path.into(),
    };

    let commit = git_short_rev_parse(&repo_dir, "HEAD")?;

    // For C++ toolchain, we'll use a date-based version format
    // Parse version from tag or use current date
    let version = if let Some(tag) = tag_or_commit {
        // Try to parse as date format YYYY.MM.DD
        if let Ok(parsed_version) = crate::distribution::parse_cpp_version(tag) {
            parsed_version
        } else {
            // Fallback to current date
            let now = Utc::now();
            Version::new(now.year() as u64, now.month() as u64, now.day() as u64)
        }
    } else {
        // Use current date as version
        let now = Utc::now();
        Version::new(now.year() as u64, now.month() as u64, now.day() as u64)
    };

    let mut version_with_commit = version.clone();
    version_with_commit.build = semver::BuildMetadata::new(&commit).unwrap();

    let dest_dir = Component::CppToolchain.get_version_dir(env, &version_with_commit);
    if dest_dir.exists() {
        return Err(RzupError::Other(format!(
            "C++ toolchain version {version_with_commit} already installed"
        )));
    }

    env.emit(RzupEvent::BuildingCppToolchainUpdate {
        message: "configuring build".into(),
    });

    // Determine the host platform for the build script
    let platform = env.platform();
    let host = match (platform.arch, platform.os) {
        ("x86_64", crate::distribution::Os::Linux) => "riscv32im-linux-x86_64",
        ("aarch64", crate::distribution::Os::MacOs) => "riscv32im-osx-arm64",
        ("aarch64", crate::distribution::Os::Linux) => "riscv32im-linux-aarch64",
        ("x86_64", crate::distribution::Os::MacOs) => "riscv32im-osx-x86_64",
        (arch, os) => {
            return Err(RzupError::Other(format!(
                "unsupported platform for C++ toolchain build: {arch} {os}"
            )));
        }
    };

    env.emit(RzupEvent::BuildingCppToolchainUpdate {
        message: format!("building toolchain for host: {}", host),
    });

    // Make the build script executable
    let build_script = repo_dir.join("build.sh");
    if build_script.exists() {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&build_script)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&build_script, perms)?;
    }

    // Run the build script
    run_command_and_stream_output("./build.sh", &[host], Some(&repo_dir), &[], |line| {
        env.emit(RzupEvent::BuildingCppToolchainUpdate {
            message: line.into(),
        });
    })
    .map_err(|e| RzupError::Other(format!("failed to build C++ toolchain: {e}")))?;

    // Install the toolchain
    env.emit(RzupEvent::BuildingCppToolchainUpdate {
        message: "installing toolchain".into(),
    });

    if let Some(parent) = dest_dir.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // The build script creates a .dist directory with the built toolchain
    let dist_dir = repo_dir.join(".dist");
    let toolchain_name = host; // Use the same name we used for the build script
    let source_toolchain_dir = dist_dir.join(&toolchain_name);

    if !source_toolchain_dir.exists() {
        return Err(RzupError::Other(format!(
            "expected toolchain directory not found: {}",
            source_toolchain_dir.display()
        )));
    }

    // Copy the built toolchain to the destination
    let toolchain_dir = dest_dir.join(&toolchain_name);
    if toolchain_dir.exists() {
        std::fs::remove_dir_all(&toolchain_dir)?;
    }

    // Copy the entire toolchain directory
    copy_dir_all(&source_toolchain_dir, &toolchain_dir)?;

    env.emit(RzupEvent::DoneBuildingCppToolchain {
        version: version_with_commit.to_string(),
    });

    Ok(version_with_commit)
}

/// Build the Groth16 toolchain from source
pub fn build_groth16_toolchain(
    env: &Environment,
    repo_url: &str,
    tag_or_commit: &Option<String>,
    path: &Option<String>,
) -> Result<Version> {
    env.emit(RzupEvent::BuildingGroth16Toolchain);

    let repo_dir = if let Some(local_path) = path {
        PathBuf::from(local_path)
    } else {
        let repo_dir = env.risc0_dir().join("tmp").join("build-groth16-toolchain");
        if repo_dir.exists() {
            std::fs::remove_dir_all(&repo_dir)?;
        }
        std::fs::create_dir_all(&repo_dir)?;

        // Clone the repository
        env.emit(RzupEvent::BuildingGroth16ToolchainUpdate {
            message: "Cloning repository...".to_string(),
        });

        let mut cmd = std::process::Command::new("git");
        cmd.arg("clone").arg(repo_url).arg(&repo_dir);
        if let Some(ref_or_commit) = tag_or_commit {
            cmd.arg("--branch").arg(ref_or_commit);
        }
        cmd.arg("--depth").arg("1");

        let output = cmd.output()?;
        if !output.status.success() {
            return Err(RzupError::Other(format!(
                "Failed to clone repository: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        repo_dir
    };

    // Change to the repository directory
    std::env::set_current_dir(&repo_dir)?;

    // Checkout specific tag or commit if provided
    if let Some(ref_or_commit) = tag_or_commit {
        env.emit(RzupEvent::BuildingGroth16ToolchainUpdate {
            message: format!("Checking out {}...", ref_or_commit),
        });

        let output = std::process::Command::new("git")
            .arg("checkout")
            .arg(ref_or_commit)
            .output()?;

        if !output.status.success() {
            return Err(RzupError::Other(format!(
                "Failed to checkout {}: {}",
                ref_or_commit,
                String::from_utf8_lossy(&output.stderr)
            )));
        }
    }

    // Get the commit hash for versioning
    let output = std::process::Command::new("git")
        .arg("rev-parse")
        .arg("HEAD")
        .output()?;

    if !output.status.success() {
        return Err(RzupError::Other(format!(
            "Failed to get commit hash: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    let commit_hash = String::from_utf8(output.stdout)
        .map_err(|e| RzupError::Other(format!("Failed to parse commit hash: {}", e)))?
        .trim()
        .to_string();
    let short_commit = &commit_hash[..8];

    // Create version with date and commit
    let now = Utc::now();
    let version_with_commit = Version::parse(&format!(
        "{}.{}.{}+{}",
        now.year(),
        now.month(),
        now.day(),
        short_commit
    ))?;

    env.emit(RzupEvent::BuildingGroth16ToolchainUpdate {
        message: "Building Groth16 toolchain...".to_string(),
    });

    // Run the Groth16 setup using xtask
    let output = std::process::Command::new("cargo")
        .arg("xtask-groth16")
        .arg("--")
        .arg(env.risc0_dir().join("tmp").join("groth16-build"))
        .output()?;

    if !output.status.success() {
        return Err(RzupError::Other(format!(
            "Failed to build Groth16 toolchain: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    // The xtask-groth16 command creates the toolchain in the specified directory
    let build_output_dir = env.risc0_dir().join("tmp").join("groth16-build");
    let version_dir = env
        .risc0_dir()
        .join("extensions")
        .join(format!("v{}-risc0-groth16", version_with_commit));

    // Create the version directory
    std::fs::create_dir_all(&version_dir)?;

    // Copy the built toolchain to the version directory
    if build_output_dir.exists() {
        copy_dir_all(&build_output_dir, &version_dir)?;
    } else {
        return Err(RzupError::Other(
            "Groth16 toolchain build output directory not found".to_string(),
        ));
    }

    env.emit(RzupEvent::DoneBuildingGroth16Toolchain {
        version: version_with_commit.to_string(),
    });

    Ok(version_with_commit)
}
