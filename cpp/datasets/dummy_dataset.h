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
#ifndef MLPERF_DATASETS_DUMMY_DATASET_H_
#define MLPERF_DATASETS_DUMMY_DATASET_H_

#include <vector>

#include "cpp/dataset.h"
#include "cpp/utils.h"

namespace mlperf {
namespace mobile {

// DummyDataset implements MLPerf's Dataset interface with randomly generated
// samples. Since it's randomly generated, it's only used for performance
// measurment without accuracy calculation.
class DummyDataset : public Dataset {
 public:
  DummyDataset(const DataFormat& input_format, const DataFormat& output_format)
      : Dataset(input_format, output_format) {
    // The number of samples only affects the randomness. Fix it to 1024.
    samples_ = std::vector<std::vector<std::vector<uint8_t>*>>(1024);
  }

  // Returns the name of the dataset.
  const std::string& Name() const override { return name_; }

  // Total number of samples in library.
  size_t TotalSampleCount() override { return samples_.size(); }

  // Loads the requested query samples into memory.
  void LoadSamplesToRam(const std::vector<QuerySampleIndex>& samples) override;

  // Unloads the requested query samples from memory.
  void UnloadSamplesFromRam(
      const std::vector<QuerySampleIndex>& samples) override;

  // GetData returns the data of a specific input.
  std::vector<void*> GetData(int sample_idx) override {
    std::vector<void*> data;
    for (std::vector<uint8_t>* v : samples_.at(sample_idx)) {
      data.push_back(v->data());
    }
    return data;
  }

  // DummyDataset has nothing to process.
  std::vector<uint8_t> ProcessOutput(
      const int sample_idx, const std::vector<void*>& outputs) override {
    return std::vector<uint8_t>();
  }

 private:
  const std::string name_ = "DummyDataset";
  // Loaded samples in RAM.
  std::vector<std::vector<std::vector<uint8_t>*>> samples_;
};

}  // namespace mobile
}  // namespace mlperf

#endif  // MLPERF_DATASETS_DUMMY_DATASET_H_
