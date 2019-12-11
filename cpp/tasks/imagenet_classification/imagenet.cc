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
#include "cpp/tasks/imagenet_classification/imagenet.h"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <streambuf>
#include <string>
#include <unordered_set>

#include "cpp/tasks/utils.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

namespace tflite {
namespace mlperf {
namespace {
// Image width of the model.
const int kImageWidth = 224;
// Image height of the model.
const int kImageHeight = 224;
}  // namespace

Imagenet::Imagenet(const std::string& image_dir, int offset, bool is_raw_images)
    : Dataset(), offset_(offset), is_raw_images_(is_raw_images) {
  std::unordered_set<std::string> exts;
  if (is_raw_images) {
    exts.insert(".rgb8");
  } else {
    exts.insert(".jpg");
    exts.insert(".jpeg");
  }
  if (evaluation::GetSortedFileNames(
          evaluation::StripTrailingSlashes(image_dir), &image_list_, exts) ==
      kTfLiteError) {
    LOG(FATAL) << "Failed to list all the images file in provided path.";
  }
  samples_ = std::vector<std::vector<data_ptr_t>>(image_list_.size());
}

void Imagenet::LoadData(int idx,
                        const std::vector<const TfLiteTensor*>& inputs) {
  if (!preprocessing_stage_) {
    evaluation::EvaluationStageConfig preprocessing_config;
    preprocessing_config.set_name("image_preprocessing");
    auto* preprocess_params = preprocessing_config.mutable_specification()
                                  ->mutable_image_preprocessing_params();
    preprocess_params->set_image_height(kImageHeight);
    preprocess_params->set_image_width(kImageWidth);
    preprocess_params->set_aspect_preserving(true);
    preprocess_params->set_output_type(static_cast<int>(inputs[0]->type));
    preprocess_params->set_load_raw_images(is_raw_images_);
    preprocessing_stage_.reset(
        new evaluation::ImagePreprocessingStage(preprocessing_config));
    if (preprocessing_stage_->Init() != kTfLiteOk) {
      LOG(FATAL) << "Failed to init preprocessing stage";
    }
  }
  // Preprocessing.
  if (idx >= image_list_.size()) {
    LOG(FATAL) << "Sample index out of bound";
  }
  std::string filename = image_list_.at(idx);
  preprocessing_stage_->SetImagePath(&filename);
  if (preprocessing_stage_->Run() != kTfLiteOk) {
    LOG(FATAL) << "Failed to run preprocessing stage";
  }

  // Move data out of preprocessing_stage_. Since the LoadSamplesToRam can load
  // multiple samples, if we don't copy the data out, it will be messed up.
  int total_size = preprocessing_stage_->GetTotalSize();
  void* data_void = preprocessing_stage_->GetPreprocessedImageData();
  if (inputs[0]->type == kTfLiteUInt8) {
    std::vector<uint8_t>* data_copy = new std::vector<uint8_t>(total_size);
    uint8_t* data = static_cast<uint8_t*>(data_void);
    std::copy(data, data + total_size, data_copy->begin());
    samples_.at(idx).push_back(data_copy);
  } else if (inputs[0]->type == kTfLiteInt8) {
    std::vector<int8_t>* data_copy = new std::vector<int8_t>(total_size);
    int8_t* data = static_cast<int8_t*>(data_void);
    std::copy(data, data + total_size, data_copy->begin());
    samples_.at(idx).push_back(data_copy);
  } else if (inputs[0]->type == kTfLiteFloat32) {
    std::vector<float>* data_copy = new std::vector<float>(total_size);
    float* data = static_cast<float*>(data_void);
    std::copy(data, data + total_size, data_copy->begin());
    samples_.at(idx).push_back(data_copy);
  } else {
    LOG(FATAL) << "Only model with type int8, uint8 and float32 is supported";
  }
}

void Imagenet::UnloadData(int idx,
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

std::vector<uint8_t> Imagenet::ProcessOutput(
    const int sample_idx, const std::vector<const TfLiteTensor*>& outputs) {
  std::vector<int32_t> topk;
  auto* output = outputs.at(0);
  if (output->type == kTfLiteUInt8) {
    topk = GetTopK(output->data.uint8, NumElements(output), 1, offset_);
  } else if (output->type == kTfLiteInt8) {
    topk = GetTopK(output->data.int8, NumElements(output), 1, offset_);
  } else if (output->type == kTfLiteFloat32) {
    topk = GetTopK(output->data.f, NumElements(output), 1, offset_);
  } else {
    LOG(FATAL) << "Only model with type int8, uint8 and float32 is supported";
  }
  // Mlperf interpret data as uint8_t* and log it as a HEX string.
  predictions_[sample_idx] = topk.at(0);
  std::vector<uint8_t> result(topk.size() * 4);
  uint8_t* temp_data = reinterpret_cast<uint8_t*>(topk.data());
  std::copy(temp_data, temp_data + result.size(), result.begin());
  return result;
}

std::string Imagenet::ComputeAccuracyString(
    const std::string& groundtruth_file) {
  // Read the imagenet val.txt file. The val.txt file contains groundtruth
  // results in ascending order. EX: the first line should be:
  // ILSVRC2012_val_00000001.JPEG 65
  std::ifstream gt_file(groundtruth_file);
  if (!gt_file.good()) {
    LOG(FATAL) << "Could not read the val file";
    return std::string("Error");
  }
  int32_t label_idx;
  std::vector<int32_t> groundtruth;
  while (gt_file >> label_idx) {
    groundtruth.push_back(label_idx);
  }

  // Read the result in mlpef log file and calculate the accuracy.
  int good = 0;
  for (auto [sample_idx, class_idx] : predictions_) {
    if (groundtruth[sample_idx] == class_idx) good++;
  }
  float accuracy = static_cast<float>(good) / predictions_.size();
  std::stringstream stream;
  stream << std::fixed << std::setprecision(2) << accuracy * 100 << "%";
  return stream.str();
}

}  // namespace mlperf
}  // namespace tflite
