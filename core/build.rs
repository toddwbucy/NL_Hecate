/// Build script: compiles CUDA kernels via nvcc when the `cuda` feature is enabled.
///
/// The cc crate invokes nvcc to compile .cu files into .o object files,
/// then links them into the final binary. These object files contain machine
/// code (PTX/SASS), NOT LLVM IR — making them opaque to AD (compiled machine code).
/// Analytical backward kernels provide gradients for these opaque regions.
///
/// Multi-architecture support: nvcc embeds SASS for sm_86/89/90a plus PTX
/// for compute_90a and compute_86 as forward-compatible fallbacks. The CUDA
/// runtime automatically selects the correct variant at kernel launch time.
///
/// Architecture targets:
///   sm_86  — Ampere (A6000, RTX 3090)
///   sm_89  — Ada Lovelace (RTX 4090)
///   sm_90a — Hopper (H100, H200) — enables TMA, cp.async pipeline
///   compute_90a PTX — forward-compat for Hopper+ (Blackwell JIT-compiles this)
///   compute_86 PTX  — legacy fallback for sm >= 86 without native SASS

fn main() {
    #[cfg(feature = "cuda")]
    {
        let cuda_path =
            std::env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda-12.8".to_string());

        cc::Build::new()
            .cuda(true)
            .cudart("shared")
            // Force optimized CUDA compilation even in debug builds.
            // nvcc -G (device debug) doubles register usage, causing "too many
            // resources requested for launch" at d >= 1536 (80+ regs × 1024
            // threads exceeds the 65536-register-per-SM limit on Ampere).
            .opt_level(2)
            .debug(false)
            // Architecture-specific SASS (native performance)
            .flag("-gencode").flag("arch=compute_86,code=sm_86")   // A6000, RTX 3090
            .flag("-gencode").flag("arch=compute_89,code=sm_89")   // RTX 4090, RTX 2000 Ada
            .flag("-gencode").flag("arch=compute_90a,code=sm_90a") // H100, H200 (TMA-enabled)
            // PTX fallback (JIT-compiled for future GPUs)
            .flag("-gencode").flag("arch=compute_90a,code=compute_90a") // Hopper+ (Blackwell)
            .flag("-gencode").flag("arch=compute_86,code=compute_86")   // Ampere+ (legacy)
            .flag("-O2")
            .flag("--use_fast_math")
            .flag("-Xcompiler")
            .flag("-Wno-unused-parameter")
            .include(format!("{}/include", cuda_path))
            .file("kernels/swa_forward.cu")
            .file("kernels/swa_backward.cu")
            .file("kernels/delta_forward.cu")
            .file("kernels/delta_backward.cu")
            .file("kernels/titans_forward.cu")
            .file("kernels/titans_backward.cu")
            .file("kernels/hebbian_forward.cu")
            .file("kernels/hebbian_backward.cu")
            .file("kernels/dgd_forward.cu")
            .file("kernels/dgd_backward.cu")
            .file("kernels/embedding.cu")
            .file("kernels/elementwise.cu")
            .file("kernels/cross_entropy.cu")
            .file("kernels/adamw.cu")
            .file("kernels/swiglu_forward.cu")
            .file("kernels/swiglu_backward.cu")
            .file("kernels/m_norm_clamp.cu")
            .file("kernels/l2_normalize.cu")
            .file("kernels/gate_backward.cu")
            .file("kernels/tnt_forward.cu")
            .file("kernels/tnt_backward.cu")
            .file("kernels/layer_norm.cu")
            .file("kernels/moneta_forward.cu")
            .file("kernels/moneta_backward.cu")
            .file("kernels/m3_optimizer.cu")
            .file("kernels/delta_chunkwise_forward.cu")
            .file("kernels/delta_chunkwise_backward.cu")
            .file("kernels/titans_chunkwise_forward.cu")
            .file("kernels/titans_chunkwise_backward.cu")
            .file("kernels/error_subtract_clip.cu")
            .file("kernels/delta_phase2_forward.cu")
            .file("kernels/delta_phase2_backward.cu")
            .file("kernels/titans_phase2_forward.cu")
            .file("kernels/titans_phase2_backward.cu")
            .file("kernels/pool_ops.cu")
            .compile("nl_hecate_cuda_kernels");

        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rerun-if-changed=kernels/swa_forward.cu");
        println!("cargo:rerun-if-changed=kernels/swa_backward.cu");
        println!("cargo:rerun-if-changed=kernels/delta_forward.cu");
        println!("cargo:rerun-if-changed=kernels/delta_backward.cu");
        println!("cargo:rerun-if-changed=kernels/titans_forward.cu");
        println!("cargo:rerun-if-changed=kernels/titans_backward.cu");
        println!("cargo:rerun-if-changed=kernels/hebbian_forward.cu");
        println!("cargo:rerun-if-changed=kernels/hebbian_backward.cu");
        println!("cargo:rerun-if-changed=kernels/dgd_forward.cu");
        println!("cargo:rerun-if-changed=kernels/dgd_backward.cu");
        println!("cargo:rerun-if-changed=kernels/embedding.cu");
        println!("cargo:rerun-if-changed=kernels/elementwise.cu");
        println!("cargo:rerun-if-changed=kernels/cross_entropy.cu");
        println!("cargo:rerun-if-changed=kernels/adamw.cu");
        println!("cargo:rerun-if-changed=kernels/swiglu_forward.cu");
        println!("cargo:rerun-if-changed=kernels/swiglu_backward.cu");
        println!("cargo:rerun-if-changed=kernels/m_norm_clamp.cu");
        println!("cargo:rerun-if-changed=kernels/l2_normalize.cu");
        println!("cargo:rerun-if-changed=kernels/gate_backward.cu");
        println!("cargo:rerun-if-changed=kernels/tnt_forward.cu");
        println!("cargo:rerun-if-changed=kernels/tnt_backward.cu");
        println!("cargo:rerun-if-changed=kernels/moneta_forward.cu");
        println!("cargo:rerun-if-changed=kernels/moneta_backward.cu");
        println!("cargo:rerun-if-changed=kernels/error_clip.cuh");
        println!("cargo:rerun-if-changed=kernels/m3_optimizer.cu");
        println!("cargo:rerun-if-changed=kernels/delta_chunkwise_forward.cu");
        println!("cargo:rerun-if-changed=kernels/delta_chunkwise_backward.cu");
        println!("cargo:rerun-if-changed=kernels/titans_chunkwise_forward.cu");
        println!("cargo:rerun-if-changed=kernels/titans_chunkwise_backward.cu");
        println!("cargo:rerun-if-changed=kernels/error_subtract_clip.cu");
        println!("cargo:rerun-if-changed=kernels/delta_phase2_forward.cu");
        println!("cargo:rerun-if-changed=kernels/delta_phase2_backward.cu");
        println!("cargo:rerun-if-changed=kernels/titans_phase2_forward.cu");
        println!("cargo:rerun-if-changed=kernels/titans_phase2_backward.cu");
        println!("cargo:rerun-if-changed=kernels/pool_ops.cu");
    }
}
