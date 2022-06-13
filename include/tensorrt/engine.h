//
// Created by fss on 22-6-10.
//

#ifndef TINYVCS_INCLUDE_TENSORRT_ENGINE_H_
#define TINYVCS_INCLUDE_TENSORRT_ENGINE_H_
#include <vector>
#include <string>
#include <memory>

#include "NvInfer.h"
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

class Trt {
 public:

  Trt();

  ~Trt();

  Trt(const Trt &trt) = delete;

  Trt &operator=(const Trt &trt) = delete;

  void EnableFP16();

  void EnableINT8();

  void SetWorkpaceSize(size_t workspaceSize);

  void SetDLACore(int dla_core);

  void SetCustomOutput(const std::vector<std::string> &custom_outputs);

  void SetLogLevel(int severity);

  void AddDynamicShapeProfile(const std::string &input_name,
                              const std::vector<int> &min_dim_vec,
                              const std::vector<int> &opt_dim_vec,
                              const std::vector<int> &maxDimVec);

  void BuildEngine(const std::string &onnx_model, const std::string &engine_file);

  bool DeserializeEngine(const std::string &engine_file, int dla_core = -1);

  bool Forward();

  bool Forward(const cudaStream_t &stream);

  void SetBindingDimensions(std::vector<int> &input_dims, int bind_index);

  void CopyFromHostToDevice(const std::vector<float> &input, int bind_index, const cudaStream_t &stream = 0);

  void CopyFromDeviceToHost(std::vector<float> &output, int bind_index, const cudaStream_t &stream = 0);

  void *GetBindingPtr(int bind_index) const; ///Set input dimension for an inference, call this before forward with dynamic shape mode.

  size_t GetBindingSize(int bind_index) const;

  nvinfer1::Dims GetBindingDims(int bind_index) const; /// get binding dimensions

  nvinfer1::DataType GetBindingDataType(int bind_index) const; ///get binding data type

  std::string GetBindingName(int bind_index) const;/// get binding name

  int GetNbInputBindings() const; ///get number of input bindings

  int GetNbOutputBindings() const; ///get number of output bindings

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

  bool is_dynamic_shape_ = false;
};
#endif //TINYVCS_INCLUDE_TENSORRT_ENGINE_H_
