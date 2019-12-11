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
#include "cpp/tasks/dummy_dataset/dummy_dataset.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <streambuf>
#include <string>
#include <unordered_set>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

namespace tflite {
namespace mlperf {

DummyDataset::DummyDataset(int num_samples, int input_size)
    : Dataset(), input_size_(input_size) {
  samples_ = std::vector<std::vector<data_ptr_t>>(num_samples);
}

void DummyDataset::LoadData(int idx,
                            const std::vector<const TfLiteTensor*>& inputs) {
  if (inputs[0]->type == kTfLiteUInt8) {
    std::vector<uint8_t>* data = new std::vector<uint8_t>(input_size_);
    for (auto it = data->begin(); it != data->end(); it++) {
      *it = random();
    }
    samples_.at(idx).push_back(data);
  } else if (inputs[0]->type == kTfLiteInt8) {
    std::vector<int8_t>* data = new std::vector<int8_t>(input_size_);
    for (auto it = data->begin(); it != data->end(); it++) {
      *it = random();
    }
    samples_.at(idx).push_back(data);
  } else if (inputs[0]->type == kTfLiteFloat32) {
    std::vector<float>* data = new std::vector<float>(input_size_);
    for (auto it = data->begin(); it != data->end(); it++) {
      *it = random();
    }
    samples_.at(idx).push_back(data);
  } else {
    LOG(FATAL) << "Only model with type int8, uint8 and float32 is supported";
  }
}

void DummyDataset::UnloadData(int idx,
                              const std::vector<const TfLiteTensor*>& inputs) {
  auto& v = samples_.at(idx);
  if (v.size() != inputs.size()) {
    LOG(FATAL) << "Number of tensor and data not matched";
  }
  for (int i = 0; i < v.size(); ++i) {
    if (inputs[i]->type == kTfLiteUInt8) {
      delete absl::get<std::vector<uint8_t>*>(v.at(i));
    } else if (inputs[i]->type == kTfLiteInt8) {
      delete absl::get<std::vector<int8_t>*>(v.at(i));
    } else if (inputs[i]->type == kTfLiteFloat32) {
      delete absl::get<std::vector<float>*>(v.at(i));
    }
  }
  v.clear();
}

std::vector<uint8_t> DummyDataset::ProcessOutput(
    const int sample_idx, const std::vector<const TfLiteTensor*>& outputs) {
  std::vector<uint8_t> result(0);
  return result;
}

}  // namespace mlperf
}  // namespace tflite
