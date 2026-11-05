pub(crate) mod batch;
pub(crate) mod metric;

#[cfg(feature = "torch-python")]
pub(crate) mod python;

use std::env;
use std::path::Path;

use log::LevelFilter;

pub struct GlobalConfig {
    pub log_level: LevelFilter,
}
impl Default for GlobalConfig {
    fn default() -> Self {
        Self {
            log_level: LevelFilter::Info,
        }
    }
}
pub fn init_globals(config: GlobalConfig) {
    env_logger::Builder::new()
        .filter_level(config.log_level)
        .target(env_logger::Target::Stdout)
        .init();

    if cfg!(feature = "torch-python") {
        let venv_path = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join(".venv");
        let py_packages_path = if cfg!(target_os = "windows") {
            venv_path.join("Lib").join("site-packages")
        } else {
            venv_path.join("lib").join("python3.12").join("site-packages")
        };
        unsafe { env::set_var("PYTHONPATH", py_packages_path.to_str().unwrap()) };
    }

    #[cfg(feature = "executorch")]
    {
        unsafe extern "C" fn cattus_executorch_emit_log(
            #[allow(unused)] timestamp: executorch_sys::executorch_timestamp_t,
            level: executorch_sys::executorch_pal_log_level,
            filename: *const ::core::ffi::c_char,
            #[allow(unused)] function: *const ::core::ffi::c_char,
            line: usize,
            message: *const ::core::ffi::c_char,
            length: usize,
        ) {
            let filename = unsafe {
                debug_assert!(!filename.is_null());
                let filename = std::ffi::CStr::from_ptr(filename);
                filename.to_str().unwrap()
            };
            let message = unsafe {
                debug_assert!(!message.is_null());
                let bytes = std::slice::from_raw_parts(message.cast(), length);
                let message = std::ffi::CStr::from_bytes_with_nul_unchecked(bytes);
                message.to_str().unwrap()
            };
            let level = match level {
                executorch_sys::executorch_pal_log_level::EXECUTORCH_PAL_LOG_LEVEL_DEBUG => log::Level::Debug,
                executorch_sys::executorch_pal_log_level::EXECUTORCH_PAL_LOG_LEVEL_INFO => log::Level::Info,
                executorch_sys::executorch_pal_log_level::EXECUTORCH_PAL_LOG_LEVEL_ERROR => log::Level::Error,
                executorch_sys::executorch_pal_log_level::EXECUTORCH_PAL_LOG_LEVEL_FATAL => log::Level::Error,
                executorch_sys::executorch_pal_log_level::EXECUTORCH_PAL_LOG_LEVEL_UNKNOWN => log::Level::Error,
            };
            let logger = log::logger();
            logger.log(
                &log::Record::builder()
                    .args(format_args!("{message}"))
                    .level(level)
                    .file(Some(filename))
                    .line(Some(line as u32))
                    .build(),
            );
            logger.flush();
        }
        unsafe {
            executorch_sys::executorch_register_pal(executorch_sys::ExecutorchPalImpl {
                init: None,
                abort: None,
                current_ticks: None,
                ticks_to_ns_multiplier: None,
                emit_log_message: Some(cattus_executorch_emit_log),
                allocate: None,
                free: None,
                source_filename: std::ptr::null(),
            });
            executorch::platform::pal_init();
        }
    }

    #[cfg(feature = "onnx-ort")]
    {
        // Since rc.13, ort only exposes an execution provider's type when its own cargo feature is
        // on, so every entry here is gated behind the matching `onnx-ort-*` feature. With none of
        // them enabled the list is empty and ONNX Runtime just uses the CPU provider.
        //
        // `commit` returns false if an environment was already configured, in which case our
        // execution provider list would be silently ignored.
        assert!(
            ort::init()
                .with_execution_providers([
                    #[cfg(feature = "onnx-ort-tensorrt")]
                    ort::ep::TensorRT::default().build(),
                    #[cfg(feature = "onnx-ort-cuda")]
                    ort::ep::CUDA::default().build(),
                    #[cfg(feature = "onnx-ort-coreml")]
                    ort::ep::CoreML::default().build(),
                ])
                .commit(),
            "the ort environment was already configured"
        );
    }
}
