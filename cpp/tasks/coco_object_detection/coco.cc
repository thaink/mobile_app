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
#include "cpp/tasks/coco_object_detection/coco.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "cpp/dataset.h"
#include "src/google/protobuf/text_format.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h"
#include "tensorflow/lite/tools/evaluation/stages/object_detection_average_precision_stage.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

namespace tflite {
namespace mlperf {
namespace {
// Image width of the model.
const int kImageWidth = 300;
// Image height of the model.
const int kImageHeight = 300;
// Number of classes in the dataset.
const int kNumClasses = 91;
}  // namespace

Coco::Coco(const std::string& image_dir, int offset, bool is_raw_images,
           const std::string& grouth_truth_pbtxt_file)
    : Dataset(), offset_(offset), is_raw_images_(is_raw_images) {
  // Read ground truth data from the pbtxt file.
  std::ifstream t(grouth_truth_pbtxt_file);
  std::string proto_str((std::istreambuf_iterator<char>(t)),
                        std::istreambuf_iterator<char>());
  evaluation::ObjectDetectionGroundTruth ground_truth_proto;
  google::protobuf::TextFormat::ParseFromString(proto_str, &ground_truth_proto);
  absl::flat_hash_map<std::string, int64_t> image_name_to_id;
  for (auto image_ground_truth : ground_truth_proto.detection_results()) {
    std::string filename = image_ground_truth.image_name();
    groundtruth_objects_[image_ground_truth.image_id()] = image_ground_truth;
    image_name_to_id[filename] = image_ground_truth.image_id();
  }
  // List images in the input directory.
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
  // Get id of listed images.
  for (const auto& image_name : image_list_) {
    std::string filename = image_name.substr(image_name.find_last_of("/") + 1);
    filename.replace(filename.find_last_of("."), std::string::npos, ".jpg");
    id_list_.push_back(image_name_to_id[filename]);
  }
  samples_ = std::vector<std::vector<data_ptr_t>>(image_list_.size());
}

void Coco::LoadData(int idx, const std::vector<const TfLiteTensor*>& inputs) {
  if (!preprocessing_stage_) {
    evaluation::EvaluationStageConfig preprocessing_config;
    preprocessing_config.set_name("image_preprocessing");
    // Expecting resized images, png format seems to give better accuracy.
    auto* preprocess_params = preprocessing_config.mutable_specification()
                                  ->mutable_image_preprocessing_params();
    preprocess_params->set_image_height(kImageHeight);
    preprocess_params->set_image_width(kImageWidth);
    preprocess_params->set_aspect_preserving(true);
    preprocess_params->set_cropping_fraction(1.0);
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

void Coco::UnloadData(int idx, const std::vector<const TfLiteTensor*>& inputs) {
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

std::vector<uint8_t> Coco::ProcessOutput(
    const int sample_idx, const std::vector<const TfLiteTensor*>& outputs) {
  int num_detections = static_cast<int>(*(outputs.at(3)->data.f));
  float* detected_label_boxes = outputs.at(0)->data.f;
  float* detected_label_indices = outputs.at(1)->data.f;
  float* detected_label_probabilities = outputs.at(2)->data.f;

  std::vector<float> data;
  evaluation::ObjectDetectionResult predict_;
  for (int i = 0; i < num_detections; ++i) {
    const int bounding_box_offset = i * 4;
    // Add for reporting to mlperf log.
    data.push_back(static_cast<float>(sample_idx));                 // Image id
    data.push_back(detected_label_boxes[bounding_box_offset + 0]);  // ymin
    data.push_back(detected_label_boxes[bounding_box_offset + 1]);  // xmin
    data.push_back(detected_label_boxes[bounding_box_offset + 2]);  // ymax
    data.push_back(detected_label_boxes[bounding_box_offset + 3]);  // xmax
    data.push_back(detected_label_probabilities[i]);                // Score
    data.push_back(detected_label_indices[i] + offset_);            // Class
    // Add for evaluation inside this class.
    auto* object = predict_.add_objects();
    auto* bbox = object->mutable_bounding_box();
    bbox->set_normalized_top(detected_label_boxes[bounding_box_offset + 0]);
    bbox->set_normalized_left(detected_label_boxes[bounding_box_offset + 1]);
    bbox->set_normalized_bottom(detected_label_boxes[bounding_box_offset + 2]);
    bbox->set_normalized_right(detected_label_boxes[bounding_box_offset + 3]);
    object->set_class_id(static_cast<int>(detected_label_indices[i]) + offset_);
    object->set_score(detected_label_probabilities[i]);
  }
  predicted_objects_[id_list_[sample_idx]] = predict_;

  // Mlperf interpret data as uint8_t* and log it as a HEX string.
  std::vector<uint8_t> result(data.size() * 4);
  uint8_t* temp_data = reinterpret_cast<uint8_t*>(data.data());
  std::copy(temp_data, temp_data + result.size(), result.begin());
  return result;
}

std::string Coco::ComputeAccuracyString(const std::string& groundtruth_file) {
  // Configs for ObjectDetectionAveragePrecisionStage.
  evaluation::EvaluationStageConfig eval_config;
  eval_config.set_name("average_precision");
  eval_config.mutable_specification()
      ->mutable_object_detection_average_precision_params()
      ->set_num_classes(kNumClasses);
  evaluation::ObjectDetectionAveragePrecisionStage eval_stage_(eval_config);

  // Init and run.
  if (eval_stage_.Init() == kTfLiteError) {
    LOG(ERROR) << "Init evaluation stage failed";
    return std::string("Error");
  }

  for (auto& [idx, objects] : predicted_objects_) {
    std::string gt_filename = groundtruth_objects_[idx].image_name();
    eval_stage_.SetEvalInputs(predicted_objects_[idx],
                              groundtruth_objects_[idx]);
    if (eval_stage_.Run() == kTfLiteError) {
      LOG(ERROR) << "Run evaluation stage failed";
      return std::string("Error");
    }
  }

  // Read the result.
  auto metrics = eval_stage_.LatestMetrics()
                     .process_metrics()
                     .object_detection_average_precision_metrics();
  std::stringstream stream;
  stream << std::fixed << std::setprecision(4)
         << metrics.overall_mean_average_precision() << " mAP";
  return stream.str();
}

}  // namespace mlperf
}  // namespace tflite
