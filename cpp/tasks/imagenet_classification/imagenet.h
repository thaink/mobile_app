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
#ifndef MLPERF_TASKS_IMAGENET_CLASSIFICATION_IMAGENET_H_
#define MLPERF_TASKS_IMAGENET_CLASSIFICATION_IMAGENET_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "cpp/dataset.h"
#include "tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h"

namespace tflite {
namespace mlperf {

class Imagenet : public Dataset {
 public:
  Imagenet(const std::string& image_dir, int offset, bool is_raw_images);

  size_t size() override { return samples_.size(); }

  // Mlperf will take care of unloading the data.
  ~Imagenet() override {}

  // Return the vector containing data coressponding to a tensor.
  // the return type is in form vector<T>*.
  data_ptr_t GetDataImpl(int sample_idx, int tensor_idx) override {
    return samples_.at(sample_idx).at(tensor_idx);
  }

  // ProcessOutput process the output data before sending to mlperf.
  std::vector<uint8_t> ProcessOutput(
      const int sample_idx,
      const std::vector<const TfLiteTensor*>& outputs) override;

  // Calculate and return the top1 accuracy.
  std::string ComputeAccuracyString(
      const std::string& groundtruth_file) override;

 private:
  // Load data of a sample to the RAM.
  void LoadData(int idx,
                const std::vector<const TfLiteTensor*>& inputs) override;

  // Delete data of a sample from RAM.
  void UnloadData(int idx,
                  const std::vector<const TfLiteTensor*>& inputs) override;

  int offset_;
  bool is_raw_images_;
  std::vector<std::string> image_list_;
  std::vector<std::vector<data_ptr_t>> samples_;
  std::unordered_map<int32_t, int32_t> predictions_;
  std::unique_ptr<evaluation::ImagePreprocessingStage> preprocessing_stage_;
};

}  // namespace mlperf
}  // namespace tflite
#endif  // MLPERF_TASKS_IMAGENET_CLASSIFICATION_IMAGENET_H_
