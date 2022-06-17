//
// Created by fss on 22-6-10.
//

#ifndef TINYVCS_INCLUDE_TENSORRT_ENGINE_H_
#define TINYVCS_INCLUDE_TENSORRT_ENGINE_H_
#include <vector>
#include <string>
#include <memory>

#include "NvInfer.h"
#include "boost/core/noncopyable.hpp"
#include "NvInferVersion.h"
#include "cuda_runtime.h"

template<typename T>
struct TrtDestroyer {
  void operator()(T *t) {
    delete t;
  }
};
template<typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer<T>>;

using Severity = nvinfer1::ILogger::Severity;
class TrtLogger : public nvinfer1::ILogger {
 public:
  void setLogSeverity(Severity severity);

 private:
  void log(Severity severity, const char *msg) noexcept override;

  Severity severity_ = Severity::kINFO;
};

void SetDevice(int device);

bool setTensorDynamicRange(const nvinfer1::INetworkDefinition &network, float in_range, float out_range);

void SaveEngine(const std::string &fileName, TrtUniquePtr<nvinfer1::IHostMemory> &plan);

class Trt : private boost::noncopyable {
 public:

  Trt();

  ~Trt();

  void EnableFP16();

  bool DeserializeEngine(const std::string &engine_file);

  bool Forward();

  void CopyFromHostToDevice(const std::vector<float> &input, int bind_index);

  void CopyFromDeviceToHost(std::vector<float> &output, int bind_index);

  nvinfer1::Dims binding_dims(int bind_index) const; /// get binding dimensions

  int input_bindings() const; ///get number of input bindings

  int output_bindings() const; ///get number of output bindings

 protected:
  void CreateDeviceBuffer();

  std::unique_ptr<TrtLogger> logger_{nullptr};

  TrtUniquePtr<nvinfer1::IBuilder> builder_{nullptr};

  TrtUniquePtr<nvinfer1::IBuilderConfig> config_{nullptr};

  TrtUniquePtr<nvinfer1::ICudaEngine> engine_{nullptr};

  TrtUniquePtr<nvinfer1::IExecutionContext> context_{nullptr};

  nvinfer1::IOptimizationProfile *profile_ = nullptr;

  std::vector<std::string> custom_outputs_;

  std::vector<void *> binding_;

  std::vector<size_t> binding_size_;

  std::vector<nvinfer1::Dims> binding_dims_;

  std::vector<nvinfer1::DataType> binding_data_type_;

  std::vector<std::string> binding_names;

  int nb_input_bindings_ = 0;

  int nb_output_bindings_ = 0;

  cudaStream_t stream_{};
};
#endif //TINYVCS_INCLUDE_TENSORRT_ENGINE_H_
