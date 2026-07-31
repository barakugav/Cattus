use ndarray::ArrayD;
use std::path::Path;

#[cfg(feature = "torch-python")]
use {crate::util::python::Unwrapy, pyo3::prelude::*};

#[cfg(feature = "onnx-tract")]
use tract_onnx::prelude::*;

#[derive(Copy, Clone, Debug, serde::Deserialize)]
pub enum TorchDevice {
    Cpu,
    Cuda,
    Mps,
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(tag = "engine")]
#[serde(rename_all = "kebab-case")]
pub enum InferenceConfig {
    OnnxOrt,
    OnnxTract,
    TorchPy { device: Option<TorchDevice> },
    TorchTchRs { device: Option<TorchDevice> },
    Executorch,
}
impl Default for InferenceConfig {
    fn default() -> Self {
        if cfg!(feature = "onnx-ort") {
            Self::OnnxOrt
        } else if cfg!(feature = "onnx-tract") {
            Self::OnnxTract
        } else if cfg!(feature = "executorch") {
            Self::Executorch
        } else if cfg!(feature = "torch-tch-rs") {
            Self::TorchTchRs { device: None }
        } else if cfg!(feature = "torch-python") {
            Self::TorchPy { device: None }
        } else {
            panic!("No model implementations available in this build");
        }
    }
}

#[allow(clippy::large_enum_variant)]
enum ModelImpl {
    #[cfg(feature = "torch-python")]
    TorchPy(Py<PyAny>),
    #[cfg(feature = "torch-tch-rs")]
    TorchRs {
        model: torch::CModule,
        device: torch::Device,
    },
    #[cfg(feature = "executorch")]
    Executorch(executorch::module::Module<'static>),
    #[cfg(feature = "onnx-tract")]
    Tract(TypedRunnableModel<TypedModel>),
    #[cfg(feature = "onnx-ort")]
    Ort {
        model: ort::session::Session,
        output_names: Vec<String>,
    },
}

pub struct Model {
    model: ModelImpl,
}
impl Model {
    pub fn new(path: impl AsRef<Path>, cfg: InferenceConfig) -> Self {
        let path = path.as_ref();
        #[allow(unused)]
        let model = match cfg {
            #[cfg(feature = "torch-python")]
            InferenceConfig::TorchPy { device } => {
                let model = Python::attach(|py| {
                    let code = cr#"
import torch
class Model:
    def __init__(self, path, device):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = torch.jit.load(path, map_location=self.device)
        self.model.eval()

    def run(self, inputs):
        with torch.no_grad():
            inputs = [torch.from_numpy(input).to(self.device) for input in inputs]
            outputs = self.model(*inputs)
            outputs = [output.detach().cpu().numpy() for output in outputs]
        return outputs
                        "#;
                    let module = PyModule::from_code(py, code, c"py/model.py", c"model").unwrapy(py);

                    let py_class = module.getattr("Model").unwrap();
                    let device = device.map(|d| match d {
                        TorchDevice::Cpu => "cpu",
                        TorchDevice::Cuda => "cuda",
                        TorchDevice::Mps => "mps",
                    });
                    py_class.call1((path, device)).unwrapy(py).into()
                });
                ModelImpl::TorchPy(model)
            }
            #[cfg(feature = "torch-tch-rs")]
            InferenceConfig::TorchTchRs { device } => {
                let device = device
                    .map(|d| match d {
                        TorchDevice::Cpu => torch::Device::Cpu,
                        TorchDevice::Cuda => torch::Device::Cuda(0),
                        TorchDevice::Mps => torch::Device::Mps,
                    })
                    .unwrap_or_else(|| {
                        if torch::utils::has_cuda() {
                            torch::Device::Cuda(0)
                        } else if torch::utils::has_mps() {
                            torch::Device::Mps
                        } else {
                            torch::Device::Cpu
                        }
                    });
                let mut model = torch::CModule::load_on_device(path, device).unwrap();
                model.set_eval();
                ModelImpl::TorchRs { model, device }
            }
            #[cfg(feature = "executorch")]
            InferenceConfig::Executorch => {
                let mut model = executorch::module::Module::from_file_path(path);
                model
                    .load(Some(executorch::program::ProgramVerification::InternalConsistency))
                    .unwrap();
                model.load_method("forward", None, None).unwrap();
                ModelImpl::Executorch(model)
            }
            #[cfg(feature = "onnx-tract")]
            InferenceConfig::OnnxTract => {
                let model = tract_onnx::onnx()
                    .model_for_path(path)
                    .unwrap()
                    .into_optimized()
                    .unwrap()
                    .into_runnable()
                    .unwrap();
                ModelImpl::Tract(model)
            }
            #[cfg(feature = "onnx-ort")]
            InferenceConfig::OnnxOrt => {
                let model = ort::session::Session::builder()
                    .unwrap()
                    .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
                    .unwrap()
                    .commit_from_file(path)
                    .unwrap();
                let output_names = model.outputs.iter().map(|o| o.name.clone()).collect();
                ModelImpl::Ort { model, output_names }
            }
            #[cfg(not(all(
                feature = "torch-python",
                feature = "torch-tch-rs",
                feature = "executorch",
                feature = "onnx-tract",
                feature = "onnx-ort"
            )))]
            unsupported_type => {
                let _ = path;
                panic!("The requested model implementation is not supported in this build: {unsupported_type:?}");
            }
        };
        #[allow(unreachable_code)]
        Self { model }
    }

    pub fn run(&mut self, inputs: Vec<ArrayD<f32>>) -> Vec<ArrayD<f32>> {
        match &mut self.model {
            #[cfg(feature = "torch-python")]
            ModelImpl::TorchPy(model) => Python::attach(|py| {
                use numpy::{IntoPyArray, PyArrayMethods};

                let inputs = inputs
                    .into_iter()
                    .map(|input| input.into_pyarray(py))
                    .collect::<Vec<_>>();
                let outputs = model.bind(py).call_method1("run", (inputs,)).unwrap();
                outputs
                    .extract::<Vec<Py<numpy::PyArrayDyn<f32>>>>()
                    .unwrap()
                    .into_iter()
                    .map(|o| o.into_bound(py).to_owned_array())
                    .collect::<Vec<_>>()
            }),
            #[cfg(feature = "torch-tch-rs")]
            ModelImpl::TorchRs { model, device } => {
                let inputs = inputs
                    .iter()
                    .map(|input| {
                        let tensor: torch::Tensor = input.try_into().unwrap();
                        torch::IValue::Tensor(tensor.to_device(*device))
                    })
                    .collect::<Vec<_>>();
                let outputs = torch::no_grad(|| model.forward_is(&inputs)).unwrap();
                let outputs = match outputs {
                    torch::IValue::Tuple(outputs) | torch::IValue::GenericList(outputs) => outputs,
                    _ => panic!("expected tuple or list as model output"),
                };
                outputs
                    .into_iter()
                    .map(|o| {
                        let torch::IValue::Tensor(tensor) = o else {
                            panic!("expected tensor as model output");
                        };
                        let array: ArrayD<f32> = (&tensor.to(torch::Device::Cpu)).try_into().unwrap();
                        array
                    })
                    .collect()
            }
            #[cfg(feature = "executorch")]
            ModelImpl::Executorch(model) => {
                let inputs = inputs
                    .into_iter()
                    .map(|input| executorch::tensor::TensorPtr::from_array(input).unwrap())
                    .collect::<Vec<_>>();
                let inputs = inputs.iter().map(executorch::evalue::EValue::from).collect::<Vec<_>>();
                let outputs = model.forward(&inputs).unwrap();
                outputs
                    .into_iter()
                    .map(|o| o.as_tensor().into_typed::<f32>().as_array().to_owned())
                    .collect()
            }
            #[cfg(feature = "onnx-tract")]
            ModelImpl::Tract(model) => {
                let inputs = TVec::from_vec(inputs.into_iter().map(Tensor::from).map(TValue::from).collect());
                let outputs = model.run(inputs).unwrap();
                outputs
                    .into_iter()
                    .map(|o| o.into_tensor().into_array::<f32>().unwrap())
                    .collect()
            }
            #[cfg(feature = "onnx-ort")]
            ModelImpl::Ort { model, output_names } => {
                let inputs = inputs
                    .into_iter()
                    .map(|input| ort::session::SessionInputValue::from(ort::value::Value::from_array(input).unwrap()))
                    .collect::<Vec<_>>();
                let inputs: &[ort::session::SessionInputValue] = &inputs;
                let mut outputs = model.run(inputs).unwrap();
                output_names
                    .iter()
                    .map(|output_name| {
                        outputs
                            .remove(output_name)
                            .unwrap()
                            .try_extract_array::<f32>()
                            .unwrap()
                            .into_owned()
                    })
                    .collect::<Vec<_>>()
            }
            #[cfg(not(any(
                feature = "torch-python",
                feature = "torch-tch-rs",
                feature = "executorch",
                feature = "onnx-tract",
                feature = "onnx-ort"
            )))]
            _ => {
                let _ = inputs;
                unreachable!()
            }
        }
    }
}
