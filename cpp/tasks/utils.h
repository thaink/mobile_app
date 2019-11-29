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
#ifndef MLPERF_TASKS_UTILS_H_
#define MLPERF_TASKS_UTILS_H_

#include <cstdint>
#include <numeric>

#include "absl/strings/ascii.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "test_settings.h"

namespace tflite {
namespace mlperf {

// Return topK indexes with highest probability.
template <typename T>
inline std::vector<int32_t> GetTopK(T* values, int num_elem, int k,
                                    int offset) {
  std::vector<int32_t> indices(num_elem - offset);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&values, offset](int a, int b) {
    return values[a + offset] > values[b + offset];
  });
  indices.resize(k);
  return indices;
}

// Convert string to mlperf::TestMode.
inline ::mlperf::TestMode Str2TestMode(const std::string& mode) {
  if (mode == "PerformanceOnly") {
    return ::mlperf::TestMode::PerformanceOnly;
  } else if (mode == "AccuracyOnly") {
    return ::mlperf::TestMode::AccuracyOnly;
  } else if (mode == "SubmissionRun") {
    return ::mlperf::TestMode::SubmissionRun;
  } else {
    LOG(ERROR) << "Mode " << mode << " is not supported.";
    return ::mlperf::TestMode::PerformanceOnly;
  }
}

// Convert string to Delegate.
inline std::vector<evaluation::TfliteInferenceParams::Delegate> Str2Delegates(
    const std::string& delegate) {
  using Delegate = evaluation::TfliteInferenceParams::Delegate;
  std::vector<Delegate> delegates;
  for (absl::string_view delegate_str : absl::StrSplit(delegate, ',')) {
    if (absl::AsciiStrToLower(delegate_str) == "gpu") {
      delegates.push_back(evaluation::TfliteInferenceParams::GPU);
    } else if (absl::AsciiStrToLower(delegate_str) == "nnapi") {
      delegates.push_back(evaluation::TfliteInferenceParams::NNAPI);
    }
  }
  return delegates;
}

}  // namespace mlperf
}  // namespace tflite
#endif  // MLPERF_TASKS_UTILS_H_
