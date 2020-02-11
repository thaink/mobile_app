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
#include "cpp/backends/tflite.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/ascii.h"
#include "cpp/backend.h"
#include "cpp/utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/stages/tflite_inference_stage.h"

namespace mlperf {
namespace mobile {
namespace {
// Convert string to Delegate.
inline tflite::evaluation::TfliteInferenceParams::Delegate Str2Delegate(
    const std::string& delegate) {
  if (absl::AsciiStrToLower(delegate) == "gpu") {
    return tflite::evaluation::TfliteInferenceParams::GPU;
  } else if (absl::AsciiStrToLower(delegate) == "nnapi") {
    return tflite::evaluation::TfliteInferenceParams::NNAPI;
  }
  return tflite::evaluation::TfliteInferenceParams::NONE;
}

// Convert TfLiteType to DataType.
inline DataType::Type TfType2DataType(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return DataType::Float32;
    case kTfLiteUInt8:
      return DataType::Uint8;
    case kTfLiteInt8:
      return DataType::Int8;
    case kTfLiteFloat16:
      return DataType::Float16;
    default:
      LOG(FATAL) << "TfLiteType " << type << " not supported";
      return DataType::Float32;
  }
}

}  // namespace

TfliteBackend::TfliteBackend(const std::string& model_file_path,
                             int num_threads, const std::string& delegate) {
  tflite::evaluation::EvaluationStageConfig inference_config;
  inference_config.set_name("inference_stage");
  auto* inference_params = inference_config.mutable_specification()
                               ->mutable_tflite_inference_params();
  inference_params->set_invocations_per_run(1);
  inference_params->set_model_file_path(model_file_path);
  inference_params->set_num_threads(num_threads);
  inference_params->set_delegate(Str2Delegate(delegate));

  inference_stage_.reset(
      new tflite::evaluation::TfliteInferenceStage(inference_config));
  if (inference_stage_->Init() != kTfLiteOk) {
    LOG(FATAL) << "Init inference stage failed";
  }
  // Collect input and output formats.
  const tflite::evaluation::TfLiteModelInfo* model_info =
      inference_stage_->GetModelInfo();
  for (const TfLiteTensor* input : model_info->inputs) {
    input_format_.emplace_back(TfType2DataType(input->type),
                               tflite::NumElements(input));
  }
  for (const TfLiteTensor* output : model_info->outputs) {
    output_format_.emplace_back(TfType2DataType(output->type),
                                tflite::NumElements(output));
  }
}

std::vector<void*> TfliteBackend::GetPredictedOutputs() {
  std::vector<void*> outputs;
  for (const TfLiteTensor* output_tensor :
       inference_stage_->GetModelInfo()->outputs) {
    switch (output_tensor->type) {
      case kTfLiteFloat32:
        outputs.push_back(output_tensor->data.f);
        break;
      case kTfLiteUInt8:
        outputs.push_back(output_tensor->data.uint8);
        break;
      case kTfLiteInt8:
        outputs.push_back(output_tensor->data.int8);
        break;
      case kTfLiteFloat16:
        outputs.push_back(output_tensor->data.f16);
        break;
      default:
        LOG(FATAL) << "Data type not yet supported";
        break;
    }
  }
  return outputs;
}

}  // namespace mobile
}  // namespace mlperf
