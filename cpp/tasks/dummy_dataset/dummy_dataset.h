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
#ifndef MLPERF_TASKS_DUMMY_DATASET_DUMMY_DATASET_H_
#define MLPERF_TASKS_DUMMY_DATASET_DUMMY_DATASET_H_

#include <string>
#include <vector>

#include "cpp/dataset.h"

namespace tflite {
namespace mlperf {

// DummyDataset implements MLPerf's Dataset interface with randomly generated
// samples. Since it's randomly generated, it's only used for performance
// measurment without accuracy calculation.
class DummyDataset : public Dataset {
 public:
  DummyDataset(int num_samples, int input_size);

  size_t size() override { return samples_.size(); }

  // Mlperf will take care of unloading the data.
  ~DummyDataset() override {}

  // ProcessOutput process the output data before sending to mlperf.
  std::vector<uint8_t> ProcessOutput(
      const int sample_idx,
      const std::vector<const TfLiteTensor*>& outputs) override;

 protected:
  // Load data of a sample to the RAM.
  void LoadData(int idx,
                const std::vector<const TfLiteTensor*>& inputs) override;

  // Delete data of a sample from RAM.
  void UnloadData(int idx,
                  const std::vector<const TfLiteTensor*>& inputs) override;

  // Return the vector containing data coressponding to a tensor.
  // the return type is in form vector<T>*.
  data_ptr_t GetDataImpl(int sample_idx, int tensor_idx) override {
    return samples_.at(sample_idx).at(tensor_idx);
  }

 private:
  int input_size_;
  std::vector<std::vector<data_ptr_t>> samples_;
};

}  // namespace mlperf
}  // namespace tflite

#endif  // MLPERF_TASKS_DUMMY_DATASET_DUMMY_DATASET_H_
