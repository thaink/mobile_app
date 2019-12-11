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
#include "cpp/mlperf_driver.h"

#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include "cpp/dataset.h"
#include "cpp/tasks/utils.h"
#include "loadgen.h"
#include "query_sample_library.h"
#include "system_under_test.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/stages/tflite_inference_stage.h"
#include "test_settings.h"

namespace tflite {
namespace mlperf {

TfliteMlperfDriver::TfliteMlperfDriver(std::string model_file_path,
                                       int num_threads, std::string delegates,
                                       int expected_input_size,
                                       int expected_output_size,
                                       std::unique_ptr<Dataset> dataset)
    : SystemUnderTest(), QuerySampleLibrary(), dataset_(std::move(dataset)) {
  evaluation::EvaluationStageConfig inference_config;
  inference_config.set_name("inference_stage");
  auto* inference_params = inference_config.mutable_specification()
                               ->mutable_tflite_inference_params();
  inference_params->set_invocations_per_run(1);
  inference_params->set_model_file_path(model_file_path);
  inference_params->set_num_threads(num_threads);
  // TODO(b/140356044) support multiple delegates.
  for (auto delegate : Str2Delegates(delegates)) {
    inference_params->set_delegate(delegate);
  }

  inference_stage_.reset(
      new evaluation::TfliteInferenceStage(inference_config));
  if (inference_stage_->Init() != kTfLiteOk) {
    LOG(FATAL) << "Init inference stage failed";
  }
  // Validate model inputs and outputs.
  const evaluation::TfLiteModelInfo* model_info =
      inference_stage_->GetModelInfo();
  if (model_info->inputs.size() != expected_input_size ||
      model_info->outputs.size() != expected_output_size) {
    LOG(ERROR) << "Model must have " << expected_input_size << " input & "
               << expected_output_size << " output";
  }
}

void TfliteMlperfDriver::IssueQuery(
    const std::vector<::mlperf::QuerySample>& samples) {
  std::vector<void*> inputs;
  auto input_type = inference_stage_->GetModelInfo()->inputs[0]->type;
  for (auto sample : samples) {
    if (input_type == kTfLiteUInt8) {
      inputs.push_back(dataset_->GetData<uint8_t>(sample.index, 0)->data());
    } else if (input_type == kTfLiteInt8) {
      inputs.push_back(dataset_->GetData<int8_t>(sample.index, 0)->data());
    } else if (input_type == kTfLiteFloat32) {
      inputs.push_back(dataset_->GetData<float>(sample.index, 0)->data());
    }
    inference_stage_->SetInputs(inputs);
    if (inference_stage_->Run() != kTfLiteOk) {
      LOG(FATAL) << "Error while inferencing model";
    }
    // Report to mlperf.
    std::vector<::mlperf::QuerySampleResponse> responses;
    std::vector<uint8_t> ans = dataset_->ProcessOutput(
        sample.index, inference_stage_->GetModelInfo()->outputs);
    responses.push_back(
        {sample.id, reinterpret_cast<std::uintptr_t>(ans.data()), ans.size()});
    ::mlperf::QuerySamplesComplete(responses.data(), responses.size());
  }
}

void TfliteMlperfDriver::StartMLPerfTest(std::string mode, int min_query_count,
                                         int min_duration,
                                         std::string output_dir) {
  // Setting the mlperf configs.
  ::mlperf::TestSettings mlperf_settings;
  mlperf_settings.scenario = ::mlperf::TestScenario::SingleStream;
  mlperf_settings.mode = Str2TestMode(mode);
  mlperf_settings.single_stream_expected_latency_ns = 1000000;
  mlperf_settings.min_query_count = min_query_count;
  mlperf_settings.min_duration_ms = min_duration;
  ::mlperf::LogSettings log_settings;
  log_settings.log_output.outdir = output_dir;

  // Start the test.
  ::mlperf::StartTest(this, this, mlperf_settings, log_settings);
}

}  // namespace mlperf
}  // namespace tflite
