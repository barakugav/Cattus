fn main() {
    if cfg!(feature = "onnx-ort") && cfg!(target_os = "macos") {
        println!("cargo:rustc-link-arg=-fapple-link-rtlib");
    }

    if cfg!(feature = "executorch") {
        println!("cargo::rerun-if-env-changed=EXECUTORCH_RS_EXECUTORCH_LIB_DIR");
        let libs_dir = std::env::var("EXECUTORCH_RS_EXECUTORCH_LIB_DIR")
            .expect("EXECUTORCH_RS_EXECUTORCH_LIB_DIR is not set, can't locate executorch static libs");
        let mut static_libs = Vec::new();
        let mut link_searchs = Vec::new();
        link_searchs.push("");

        link_searchs.push("/kernels/portable/");
        static_libs.push("portable_kernels");
        // static_libs.push("portable_ops_lib");

        link_searchs.push("/configurations/");
        static_libs.push("optimized_native_cpu_ops_lib");

        link_searchs.push("/kernels/optimized/");
        static_libs.push("optimized_kernels");
        static_libs.push("eigen_blas");
        static_libs.push("cpublas");

        link_searchs.push("/extension/threadpool/");
        static_libs.push("extension_threadpool");

        link_searchs.push("/backends/xnnpack/third-party/cpuinfo/");
        static_libs.push("cpuinfo");

        link_searchs.push("/backends/xnnpack/third-party/pthreadpool/");
        static_libs.push("pthreadpool");

        if rerun_env("CATTUS_MPS").as_deref() == Some("1") {
            link_searchs.push("/backends/apple/mps/");
            static_libs.push("mpsdelegate");

            // println!(
            //     "cargo::rustc-link-search=native=/Library/Developer/CommandLineTools/usr/lib/clang/16/lib/darwin/"
            // );
            // println!("cargo::rustc-link-lib=static=clang_rt.osx");

            println!("cargo::rustc-link-lib=dylib=objc");
            println!("cargo::rustc-link-lib=framework=Metal");
            println!("cargo::rustc-link-lib=framework=MetalPerformanceShaders");
            println!("cargo::rustc-link-lib=framework=MetalPerformanceShadersGraph");
            println!("cargo::rustc-link-arg=Accelerate");
            println!("cargo::rustc-link-lib=framework=Foundation");
            println!("cargo::rustc-link-lib=framework=CoreGraphics");
            // println!("cargo:rustc-link-lib=dylib=System");
        }

        if rerun_env("CATTUS_XNNPACK").as_deref() == Some("1") {
            link_searchs.push("/backends/xnnpack/");
            link_searchs.push("/backends/xnnpack/third-party/XNNPACK/");
            static_libs.push("xnnpack_backend");
            static_libs.push("XNNPACK");
            static_libs.push("xnnpack-microkernels-prod");

            if cfg!(target_arch = "arm") || cfg!(target_arch = "aarch64") {
                link_searchs.push("/kleidiai/");
                static_libs.push("kleidiai");
            }
        }

        for link_search in link_searchs {
            println!("cargo::rustc-link-search=native={libs_dir}{link_search}");
        }
        for link_lib in static_libs {
            println!("cargo::rustc-link-lib=static:+whole-archive={link_lib}");
        }
    }
}

fn rerun_env(env_var: &str) -> Option<String> {
    println!("cargo::rerun-if-env-changed={env_var}");
    std::env::var(env_var).ok()
}
