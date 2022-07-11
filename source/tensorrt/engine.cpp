//
// Created by fss on 22-6-10.
//
#include <memory>
#include <fstream>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include "boost/filesystem.hpp"
#include "glog/logging.h"
#include "cuda/cuda_utils.h"
#include "tensorrt/engine.h"
#include "fmt/core.h"

void SetDevice(int device) {
  LOG(INFO) << fmt::format("set device {}", device);
  CUDA_CHECK(cudaSetDevice(device));
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
  SetDevice(0);
  cudaStreamCreate(&stream_);
  LOG(INFO) << "Create trt instance";
  logger_ = std::make_unique<TrtLogger>();

  builder_.reset(nvinfer1::createInferBuilder(*logger_));
  LOG_IF(FATAL, builder_ == nullptr) << "create trt builder failed";

  config_.reset(builder_->createBuilderConfig());
  LOG_IF(FATAL, config_ == nullptr) << "create trt builder config failed";

  config_->setMaxWorkspaceSize((1 << 30)); // 1GB
  profile_ = builder_->createOptimizationProfile();
  LOG_IF(FATAL, profile_ == nullptr) << "create trt builder optimazation profile failed";
}

Trt::~Trt() {
  LOG(INFO) << "destroy Trt instance";

  profile_ = nullptr;
  for (auto &i : binding_) {
    safeCudaFree(i);
  }
  if (stream_) {
    cudaStreamDestroy(stream_);
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

bool Trt::DeserializeEngine(const std::string &engine_file) {
  std::ifstream in(engine_file.c_str(), std::ifstream::binary);
  if (in.is_open()) {
    LOG(INFO) << "deserialize engine from " << engine_file;
    size_t buf_count = boost::filesystem::file_size(engine_file);
    std::unique_ptr<char[]> engine_buf(new char[buf_count]);
    in.read(engine_buf.get(), buf_count);
    TrtUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(*logger_)};

    engine_.reset(runtime->deserializeCudaEngine((void *) engine_buf.get(), buf_count));
    LOG_IF(FATAL, engine_ == nullptr) << "engine create failed";
    context_.reset(engine_->createExecutionContext());
    LOG_IF(FATAL, context_ == nullptr) << "context create faild";
    CreateDeviceBuffer();
    return true;
  }
  return false;
}

bool Trt::Forward() {
  return context_->enqueueV2(&binding_[0], stream_, nullptr);
}

void Trt::CopyFromHostToDevice(const std::vector<float> &input,
                               int bind_index) {
  CUDA_CHECK(cudaMemcpyAsync(binding_[bind_index], input.data(),
                             input.size() * sizeof(float), cudaMemcpyHostToDevice, stream_))
}

void Trt::CopyFromDeviceToDevice(float *input, size_t size,
                                 int bind_index) {
  CUDA_CHECK(cudaMemcpyAsync(binding_[bind_index], input,
                             size * sizeof(float), cudaMemcpyDeviceToDevice, stream_))
}

void Trt::CopyFromDeviceToHost(std::vector<float> &output, int bind_index) {
  CUDA_CHECK(cudaMemcpyAsync(output.data(), binding_[bind_index],
                             output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_))
}

void Trt::CopyFromDeviceToDevice2(float *outputs, int elements_size, int bind_index) const {
  CHECK(outputs != nullptr);
  CUDA_CHECK(cudaMemcpyAsync(outputs, binding_[bind_index],
                             elements_size * sizeof(float), cudaMemcpyDeviceToDevice, stream_))
}

nvinfer1::Dims Trt::binding_dims(int bind_index) const {
  return binding_dims_[bind_index];
}

int Trt::input_bindings() const {
  return nb_input_bindings_;
}

int Trt::output_bindings() const {
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
    dims = engine_->getBindingDimensions(i);
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
    if (engine_->bindingIsInput(i)) {
      binding_[i] = safeCudaMalloc(totalSize);
      nb_input_bindings_ = i;
    } else {
      binding_[i] = safeCudaMalloc(totalSize);
      nb_output_bindings_ = i;
    }
  }
}

