/* Copyright 2020 The MLPerf Authors. All Rights Reserved.

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
#include "cpp/datasets/ade20k.h"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <streambuf>
#include <string>
#include <unordered_set>

#include "cpp/utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

namespace mlperf {
namespace mobile {
namespace {
inline TfLiteType DataType2TfType(DataType::Type type) {
  switch (type) {
    case DataType::Float32:
      return kTfLiteFloat32;
    case DataType::Uint8:
      return kTfLiteUInt8;
    case DataType::Int8:
      return kTfLiteInt8;
    case DataType::Float16:
      return kTfLiteFloat16;
  }
  return kTfLiteNoType;
}
}  // namespace

ADE20K::ADE20K(const DataFormat& input_format, const DataFormat& output_format,
               const std::string& image_dir,
               const std::string& ground_truth_dir, int num_classes,
               int image_width, int image_height)
    : Dataset(input_format, output_format),
      num_classes_(num_classes),
      image_width_(image_width),
      image_height_(image_height) {
  if (input_format_.size() != 1 || output_format_.size() != 1) {
    LOG(FATAL) << "ADE20K model only supports 1 input and 1 output";
    return;
  }
  // Finds all images under image_dir.
  std::unordered_set<std::string> exts{".rgb8", ".jpg", ".jpeg"};
  TfLiteStatus ret = tflite::evaluation::GetSortedFileNames(
      tflite::evaluation::StripTrailingSlashes(image_dir), &image_list_, exts);
  if (ret == kTfLiteError || image_list_.empty()) {
    LOG(FATAL) << "Failed to list all the images file in provided path";
    return;
  }
  samples_ =
      std::vector<std::vector<std::vector<uint8_t>*>>(image_list_.size());
  // Finds all ground truth files under ground_truth_dir.
  std::unordered_set<std::string> gt_exts{".raw"};
  ret = tflite::evaluation::GetSortedFileNames(
      tflite::evaluation::StripTrailingSlashes(ground_truth_dir),
      &ground_truth_list_, gt_exts);
  if (ret == kTfLiteError || ground_truth_list_.empty()) {
    LOG(ERROR) << "Failed to list all the ground truth files in provided path. "
                  "Only measuring performance.";
  }

  // Prepares the preprocessing stage.
  tflite::evaluation::ImagePreprocessingConfigBuilder builder(
      "image_preprocessing", DataType2TfType(input_format_.at(0).type));
  builder.AddDefaultNormalizationStep();
  preprocessing_stage_.reset(
      new tflite::evaluation::ImagePreprocessingStage(builder.build()));
  if (preprocessing_stage_->Init() != kTfLiteOk) {
    LOG(FATAL) << "Failed to init preprocessing stage";
  }
}

void ADE20K::LoadSamplesToRam(const std::vector<QuerySampleIndex>& samples) {
  for (QuerySampleIndex sample_idx : samples) {
    // Preprocessing.
    if (sample_idx >= image_list_.size()) {
      LOG(FATAL) << "Sample index out of bound";
    }
    std::string filename = image_list_.at(sample_idx);
    preprocessing_stage_->SetImagePath(&filename);
    if (preprocessing_stage_->Run() != kTfLiteOk) {
      LOG(FATAL) << "Failed to run preprocessing stage";
    }

    // Move data out of preprocessing_stage_ so it can be reused.
    int total_byte = input_format_[0].size * input_format_[0].GetByte();
    void* data_void = preprocessing_stage_->GetPreprocessedImageData();
    std::vector<uint8_t>* data_uint8 = new std::vector<uint8_t>(total_byte);
    std::copy(static_cast<uint8_t*>(data_void),
              static_cast<uint8_t*>(data_void) + total_byte,
              data_uint8->begin());
    samples_.at(sample_idx).push_back(data_uint8);
  }
}

void ADE20K::UnloadSamplesFromRam(
    const std::vector<QuerySampleIndex>& samples) {
  for (QuerySampleIndex sample_idx : samples) {
    for (std::vector<uint8_t>* v : samples_.at(sample_idx)) {
      delete v;
    }
    samples_.at(sample_idx).clear();
  }
}

std::vector<uint8_t> ADE20K::ProcessOutput(const int sample_idx,
                                           const std::vector<void*>& outputs) {
  if (ground_truth_list_.empty()) {
    return std::vector<uint8_t>();
  }
  std::vector<uint64_t> tps, fps, fns;
  std::string filename = ground_truth_list_.at(sample_idx);
  std::ifstream stream(filename, std::ios::in | std::ios::binary);
  std::vector<uint8_t> ground_truth_vector(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());

  auto output = reinterpret_cast<int32_t*>(outputs[0]);

  for (int c = 1; c <= num_classes_; c++) {
    uint64_t true_positive = 0, false_positive = 0, false_negative = 0;

    for (int i = 0; i < (image_width_ * image_height_ - 1); i++) {
      auto p = (uint8_t)0x000000ff & output[i];
      auto g = ground_truth_vector[i];

      // trichotomy
      if ((p == c) or (g == c)) {
        if (p == g) {
          true_positive++;
        } else if (p == c) {
          if ((g > 0) && (g <= num_classes_)) false_positive++;
        } else {
          false_negative++;
        }
      }
    }

    tps.push_back(true_positive);
    fps.push_back(false_positive);
    fns.push_back(false_negative);
  }

  if (!initialized_) {
    initialized_ = true;

    for (int j = 0; j < num_classes_; j++) {
      tp_acc_.push_back(tps[j]);
      fp_acc_.push_back(fps[j]);
      fn_acc_.push_back(fns[j]);
    }
  } else {
    for (int j = 0; j < num_classes_; j++) {
      tp_acc_[j] += tps[j];
      fp_acc_[j] += fps[j];
      fn_acc_[j] += fns[j];
    }
  }

#if __DEBUG__
  for (int j = 0; j < num_classes_; j++) {
    LOG(INFO) << tp_acc_[j] << ", " << fp_acc_[j] << ", " << fn_acc_[j];
    if (j < 30) std::cout << ", ";
  }
  LOG(INFO) << "\n";
  for (int j = 0; j < num_classes_; j++) {
    LOG(INFO) << "mIOU class " << j + 1 << ": "
              << tp_acc_[j] * 1.0 / (tp_acc_[j] + fp_acc_[j] + fn_acc_[j])
              << "\n";
  }
#endif

  return std::vector<uint8_t>();
}

float ADE20K::ComputeAccuracy() {
  if (ground_truth_list_.empty()) {
    return 0.0;
  }
  float iou_sum = 0.0;
  for (int j = 0; j < num_classes_; j++) {
    auto iou = tp_acc_[j] * 1.0 / (tp_acc_[j] + fp_acc_[j] + fn_acc_[j]);
#if __DEBUG__
    LOG(INFO) << "IOU class " << j + 1 << ": " << tp_acc_[j] << ", "
              << fp_acc_[j] << ", " << fn_acc_[j] << ", " << iou << "\n";
#endif
    iou_sum += iou;
  }
#if __DEBUG__
  LOG(INFO) << "mIOU over_all: " << iou_sum / num_classes_ << "\n";
#endif
  return iou_sum / num_classes_;
}

std::string ADE20K::ComputeAccuracyString() {
  float result = ComputeAccuracy();
  if (result == 0.0f) {
    return std::string("N/A");
  }
  std::stringstream stream;
  stream << std::fixed << std::setprecision(4) << result << " mIoU";

  return stream.str();
}

}  // namespace mobile
}  // namespace mlperf
