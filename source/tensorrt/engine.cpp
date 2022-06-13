//
// Created by fss on 22-6-10.
//
#include <memory>
#include <fstream>
#include <cassert>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include "glog/logging.h"
#include "cu_utils.h"
#include "tensorrt/engine.h"
#include "fmt/core.h"

void SetDevice(int device) {
  LOG(INFO) << fmt::format("set device {}", device);
  CUDA_CHECK(cudaSetDevice(device));
}

int GetDevice() {
  int device = -1;
  CUDA_CHECK(cudaGetDevice(&device));
  if (device != -1) {
    return device;
  } else {
    LOG(ERROR) << "get device error";
    return -1;
  }
}

void SaveEngine(const std::string &fileName, TrtUniquePtr<nvinfer1::IHostMemory> &plan) {
  if (fileName.empty()) {
    LOG(WARNING) << "empty engine file name, skip save";
    return;
  }
  LOG_IF(FATAL, !plan) << "plan is empty";

  LOG(INFO) << "save engine to " << fileName;
  std::ofstream file;
  file.open(fileName, std::ios::binary | std::ios::out);
  if (!file.is_open()) {
    LOG(ERROR) << fmt::format("read create engine file {} failed", fileName);
    return;
  }
  file.write((const char *) plan->data(), plan->size());
  file.close();
}

bool setTensorDynamicRange(const nvinfer1::INetworkDefinition &network, float in_range, float out_range) {
  for (int l = 0; l < network.getNbLayers(); l++) {
    auto *layer = network.getLayer(l);
    for (int i = 0; i < layer->getNbInputs(); i++) {
      nvinfer1::ITensor *input{layer->getInput(i)};
      if (input && !input->dynamicRangeIsSet()) {
        if (!input->setDynamicRange(-in_range, in_range)) {
          return false;
        }
      }
    }
    for (int o = 0; o < layer->getNbOutputs(); o++) {
      nvinfer1::ITensor *output{layer->getOutput(o)};
      if (output && !output->dynamicRangeIsSet()) {
        if (layer->getType() == nvinfer1::LayerType::kPOOLING) {
          if (!output->setDynamicRange(-in_range, in_range)) {
            return false;
          }
        } else {
          if (!output->setDynamicRange(-out_range, out_range)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

void TrtLogger::log(Severity severity, const char *msg) noexcept {
  if (severity <= severity_) {
    switch (severity) {
      case Severity::kINTERNAL_ERROR: {
        LOG(FATAL) << msg;
        break;
      }
      case Severity::kERROR: {
        LOG(ERROR) << msg;
        break;
      }
      case Severity::kWARNING: {
        LOG(WARNING) << msg;
        break;
      }
      case Severity::kINFO:
      case Severity::kVERBOSE: {
        LOG(INFO) << msg;
        break;
      }
      default:break;
    }
  }
}

void TrtLogger::setLogSeverity(Severity severity) {
  severity_ = severity;
}

Trt::Trt() {
  LOG(INFO) << "Create trt instance";
  logger_ = std::make_unique<TrtLogger>();
  initLibNvInferPlugins(logger_.get(), "");

  builder_.reset(nvinfer1::createInferBuilder(*logger_));
  LOG_IF(FATAL, builder_ == nullptr) << "create trt builder failed";

  config_.reset(builder_->createBuilderConfig());
  LOG_IF(FATAL, config_ == nullptr) << "create trt builder config failed";

  config_->setMaxWorkspaceSize(1 << 30); // 1GB
  profile_ = builder_->createOptimizationProfile();
  LOG_IF(FATAL, profile_ == nullptr) << "create trt builder optimazation profile failed";
}

Trt::~Trt() {
  LOG(INFO) << "destroy Trt instance";

  profile_ = nullptr;
  for (size_t i = 0; i < binding_.size(); i++) {
    safeCudaFree(binding_[i]);
  }
}

void Trt::EnableFP16() {
  LOG_IF(FATAL, !builder_ && !config_) << "please set config before build engine";
  LOG(INFO) << "enable FP16";

  if (!builder_->platformHasFastFp16()) {
    LOG(WARNING) << "the platform doesn't have native fp16 support";
  }
  config_->setFlag(nvinfer1::BuilderFlag::kFP16);
}

void Trt::EnableINT8() {
  LOG_IF(FATAL, !builder_ && !config_) << "please set config before build engine";
  LOG(INFO) << "enable int8, call SetInt8Calibrator to set int8 calibrator";
  if (!builder_->platformHasFastInt8()) {
    LOG(WARNING) << "the platform doesn't have native int8 support";
  }
  config_->setFlag(nvinfer1::BuilderFlag::kINT8);
}

void Trt::SetWorkpaceSize(size_t workspaceSize) {
  LOG_IF(FATAL, !builder_ && !config_) << "please set config before build engine";
  config_->setMaxWorkspaceSize(workspaceSize);
  LOG(INFO) << fmt::format("set max workspace size: {}", config_->getMaxWorkspaceSize());
}

void Trt::SetDLACore(int dla_core) {
  LOG_IF(FATAL, !builder_ && !config_) << "please set config before build engine";
  LOG(INFO) << fmt::format("set dla core {}", dla_core);
  if (dla_core >= 0) {
    config_->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
    config_->setDLACore(dla_core);
    config_->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
  }
}

void Trt::SetCustomOutput(const std::vector<std::string> &custom_outputs) {
  LOG_IF(FATAL, !builder_ && !config_) << "please set config before build engine";
  LOG(INFO) << "set custom output";
  custom_outputs_ = custom_outputs;
}

void Trt::SetLogLevel(int severity) {
  LOG(INFO) << fmt::format("set log level {}", severity);
  logger_->setLogSeverity(static_cast<nvinfer1::ILogger::Severity>(severity));
}

void Trt::AddDynamicShapeProfile(const std::string &input_name,
                                 const std::vector<int> &min_dim_vec,
                                 const std::vector<int> &opt_dim_vec,
                                 const std::vector<int> &maxDimVec) {
  LOG(INFO) << fmt::format("add profile for {}", input_name);
  nvinfer1::Dims minDim, optDim, maxDim;
  int nbDims = opt_dim_vec.size();
  minDim.nbDims = nbDims;
  optDim.nbDims = nbDims;
  maxDim.nbDims = nbDims;
  for (int i = 0; i < nbDims; i++) {
    minDim.d[i] = min_dim_vec[i];
    optDim.d[i] = opt_dim_vec[i];
    maxDim.d[i] = maxDimVec[i];
  }
  profile_->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, minDim);
  profile_->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, optDim);
  profile_->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, maxDim);
  is_dynamic_shape_ = true;
}

void Trt::BuildEngine(
    const std::string &onnx_model,
    const std::string &engine_file) {
  LOG(INFO) << fmt::format("build onnx engine from {}...", onnx_model);

  TrtUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(*logger_)};
  LOG_IF(FATAL, !runtime) << "create trt runtime failed";

  auto flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  TrtUniquePtr<nvinfer1::INetworkDefinition> network{builder_->createNetworkV2(flag)};
  LOG_IF(FATAL, !network) << "create trt network failed";

  TrtUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, *logger_)};
  LOG_IF(FATAL, !network) << "create trt onnx parser failed";
  bool parse_success = parser->parseFromFile(onnx_model.c_str(),
                                             static_cast<int>(Severity::kWARNING));

  LOG_IF(FATAL, !parse_success) << "parse onnx file failed";
  if (!custom_outputs_.empty()) {

  }

  if (config_->getFlag(nvinfer1::BuilderFlag::kINT8) && config_->getInt8Calibrator() == nullptr) {
    LOG(WARNING) << "No calibrator found, using fake scale";
    setTensorDynamicRange(*network, 2.0f, 4.0f);
  }

  if (is_dynamic_shape_) {
    LOG_IF(FATAL, !profile_->isValid()) << "invalid dynamic shape profile";
    config_->addOptimizationProfile(profile_);
  }

  TrtUniquePtr<nvinfer1::IHostMemory> plan{builder_->buildSerializedNetwork(*network, *config_)};
  engine_.reset(runtime->deserializeCudaEngine(plan->data(), plan->size()));
  LOG_IF(FATAL, !engine_) << "build trt engine failed";

  SaveEngine(engine_file, plan);
  context_.reset(engine_->createExecutionContext());
  LOG_IF(FATAL, !context_) << "create execution context failed";

  CreateDeviceBuffer();
  builder_.reset(nullptr);
  config_.reset(nullptr);
}

bool Trt::DeserializeEngine(const std::string &engine_file, int dla_core) {
  std::ifstream in(engine_file.c_str(), std::ifstream::binary);
  if (in.is_open()) {
    LOG(INFO) << "deserialize engine from " << engine_file;
    auto const start_pos = in.tellg();
    in.ignore(std::numeric_limits<std::streamsize>::max());
    size_t buf_count = in.gcount();
    in.seekg(start_pos);
    std::unique_ptr<char[]> engine_buf(new char[buf_count]);
    in.read(engine_buf.get(), buf_count);
    TrtUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(*logger_)};
    if (dla_core >= 0) {
      runtime->setDLACore(dla_core);
    }
    engine_.reset(runtime->deserializeCudaEngine((void *) engine_buf.get(), buf_count));
    LOG_IF(FATAL, engine_ == nullptr) << "engine create failed";
    context_.reset(engine_->createExecutionContext());
    LOG_IF(FATAL, context_ == nullptr) << "context create faild";
    if (is_dynamic_shape_) {
      LOG_IF(FATAL, !profile_->isValid()) << "invalid dynamic shape profile";
      config_->addOptimizationProfile(profile_);
    }
    CreateDeviceBuffer();
    return true;
  }
  return false;
}

bool Trt::Forward() {
  return context_->executeV2(&binding_[0]);
}

bool Trt::Forward(const cudaStream_t &stream) {
  return context_->enqueueV2(&binding_[0], stream, nullptr);
}

void Trt::SetBindingDimensions(std::vector<int> &input_dims, int bind_index) {
  nvinfer1::Dims dims;
  int nbDims = input_dims.size();
  dims.nbDims = nbDims;
  for (int i = 0; i < nbDims; i++) {
    dims.d[i] = input_dims[i];
  }
  context_->setBindingDimensions(bind_index, dims);
}

void Trt::CopyFromHostToDevice(const std::vector<float> &input,
                               int bind_index, const cudaStream_t &stream) {
  LOG(INFO) << fmt::format("input size: {}, binding size: {}", input.size() * sizeof(float), binding_size_[bind_index]);
  CUDA_CHECK(cudaMemcpyAsync(binding_[bind_index], input.data(),
                             input.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
}

void Trt::CopyFromDeviceToHost(std::vector<float> &output, int bind_index,
                               const cudaStream_t &stream) {
  CUDA_CHECK(cudaMemcpyAsync(output.data(), binding_[bind_index],
                             output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
}

void *Trt::GetBindingPtr(int bind_index) const {
  return binding_[bind_index];
}

size_t Trt::GetBindingSize(int bind_index) const {
  return binding_size_[bind_index];
}

nvinfer1::Dims Trt::GetBindingDims(int bind_index) const {
  return binding_dims_[bind_index];
}

nvinfer1::DataType Trt::GetBindingDataType(int bind_index) const {
  return binding_data_type_[bind_index];
}

std::string Trt::GetBindingName(int bind_index) const {
  return binding_names[bind_index];
}

int Trt::GetNbInputBindings() const {
  return nb_input_bindings_;
}

int Trt::GetNbOutputBindings() const {
  return nb_output_bindings_;
}

void Trt::CreateDeviceBuffer() {
  LOG(INFO) << "malloc device memory";
  int nbBindings = engine_->getNbBindings();
  LOG(INFO) << "nbBindings: " << nbBindings;
  binding_.resize(nbBindings);
  binding_size_.resize(nbBindings);
  binding_names.resize(nbBindings);
  binding_dims_.resize(nbBindings);
  binding_data_type_.resize(nbBindings);
  for (int i = 0; i < nbBindings; i++) {
    const char *name = engine_->getBindingName(i);
    nvinfer1::DataType dtype = engine_->getBindingDataType(i);
    nvinfer1::Dims dims;
    if (is_dynamic_shape_) {
      // specify max input dimensions to get max output dimensions
      if (engine_->bindingIsInput(i)) {
        dims = profile_->getDimensions(name, nvinfer1::OptProfileSelector::kMAX);
        context_->setBindingDimensions(i, dims);
      } else {
        assert(context_->allInputDimensionsSpecified());
        dims = context_->getBindingDimensions(i);
      }
    } else {
      dims = engine_->getBindingDimensions(i);
    }
    int64_t totalSize = volume(dims) * getElementSize(dtype);
    binding_size_[i] = totalSize;
    binding_names[i] = name;
    binding_dims_[i] = dims;
    binding_data_type_[i] = dtype;
    if (engine_->bindingIsInput(i)) {
      LOG(INFO) << "input: ";
    } else {
      LOG(INFO) << "output: ";
    }
    LOG(INFO) << fmt::format("binding bindIndex: {}, name: {}, size in byte: {}", i, name, totalSize);
    LOG(INFO) << fmt::format("binding dims with {} dimension", dims.nbDims);
    for (int j = 0; j < dims.nbDims; j++) {
      LOG(INFO) << dims.d[j] << " x ";
    }
    LOG(INFO) << "\b\b  ";
    binding_[i] = safeCudaMalloc(totalSize);
    if (engine_->bindingIsInput(i)) {
      nb_input_bindings_++;
    } else {
      nb_output_bindings_++;
    }
  }
}

