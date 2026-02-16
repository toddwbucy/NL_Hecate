/// Build script: compiles CUDA kernels via nvcc when the `cuda` feature is enabled.
///
/// The cc crate invokes nvcc to compile .cu files into .o object files,
/// then links them into the final binary. These object files contain machine
/// code (PTX/SASS), NOT LLVM IR — making them opaque to Enzyme's AD pass.
/// This is the barrier pattern from Phase 0 Finding #3.

fn main() {
    #[cfg(feature = "cuda")]
    {
        let cuda_path =
            std::env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda-12.8".to_string());

        cc::Build::new()
            .cuda(true)
            .cudart("shared")
            .flag("-gencode")
            .flag("arch=compute_86,code=sm_86")
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
            .compile("nl_hecate_cuda_kernels");

        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rerun-if-changed=kernels/swa_forward.cu");
        println!("cargo:rerun-if-changed=kernels/swa_backward.cu");
        println!("cargo:rerun-if-changed=kernels/delta_forward.cu");
        println!("cargo:rerun-if-changed=kernels/delta_backward.cu");
        println!("cargo:rerun-if-changed=kernels/titans_forward.cu");
        println!("cargo:rerun-if-changed=kernels/titans_backward.cu");
        println!("cargo:rerun-if-changed=kernels/hebbian_forward.cu");
        println!("cargo:rerun-if-changed=kernels/hebbian_backward.cu");
    }
}
