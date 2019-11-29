/* Copyright 2019 The MLPerf Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef MLPERF_DATASET_H_
#define MLPERF_DATASET_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "absl/types/variant.h"
#include "query_sample_library.h"
#include "tensorflow/lite/c/common.h"
namespace tflite {
namespace mlperf {

// Type data_ptr_t can hold multiple types for passing data around.
using data_ptr_t = absl::variant<std::vector<uint8_t>*, std::vector<int8_t>*,
                                 std::vector<float>*>;

// Dataset is an interface adapting different datasets to use
// in TFLRunner. Each dataset should implement their methods to load,
// pre-process data and post-process the output data.
class Dataset {
 public:
  Dataset() {}

  virtual ~Dataset() {}

  // Return the number of samples in the dataset.
  virtual size_t size() = 0;

  // GetData return the data of a specific tensor.
  template <typename T>
  std::vector<T>* GetData(int sample_idx, int tensor_idx) {
    return absl::get<std::vector<T>*>(GetDataImpl(sample_idx, tensor_idx));
  }

  // ProcessOutput process the output data before sending to mlperf.
  // This function only get called on Accuracy mode so we don't need to care
  // about its performance.
  virtual std::vector<uint8_t> ProcessOutput(
      const int sample_idx,
      const std::vector<const TfLiteTensor*>& outputs) = 0;

  // Calculate the accuracy from mlperf logfile and groundtruth file.
  // Implementing this function is optional, you don't need to implement
  // it if you want to use other scripts for accuracy calculation.
  // The result is a string since different metrics have different formats.
  virtual std::string ComputeAccuracyString(
      const std::string& groundtruth_file) {
    return std::string("N/A");
  }

  // Loads the requested query samples into memory. The list of input tensors
  // are only used to refer tensor type.
  void LoadSamplesToRam(const std::vector<::mlperf::QuerySampleIndex>& samples,
                        const std::vector<const TfLiteTensor*>& inputs) {
    for (auto sample_index : samples) {
      LoadData(sample_index, inputs);
    }
  }

  // Unloads the requested query samples from memory. The list of input tensors
  // are only used to refer tensor type.
  void UnloadSamplesFromRam(
      const std::vector<::mlperf::QuerySampleIndex>& samples,
      const std::vector<const TfLiteTensor*>& inputs) {
    for (auto sample_index : samples) {
      UnloadData(sample_index, inputs);
    }
  }

 protected:
  // Load data of a sample to the RAM.
  virtual void LoadData(int idx,
                        const std::vector<const TfLiteTensor*>& inputs) = 0;

  // Delete data of a sample from RAM.
  virtual void UnloadData(int idx,
                          const std::vector<const TfLiteTensor*>& inputs) = 0;

  // Returns the vector containing data coressponding to a tensor.
  virtual data_ptr_t GetDataImpl(int sample_idx, int tensor_idx) = 0;
};

}  // namespace mlperf
}  // namespace tflite
#endif  // MLPERF_DATASET_H_
