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
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "cpp/mlperf_driver.h"
#include "cpp/tasks/dummy_dataset/dummy_dataset.h"
#include "tensorflow/lite/tools/command_line_flags.h"

namespace tflite {
namespace mlperf {

constexpr char kModelFileFlag[] = "model_file";
constexpr char kInterpreterThreadsFlag[] = "num_threads";
constexpr char kDelegateFlag[] = "delegate";
constexpr char kOutputDirFlag[] = "output_dir";
constexpr char kMinCountFlag[] = "min_query_count";
constexpr char kMinDurationFlag[] = "min_duration";
constexpr char kSampleSize[] = "sample_size";
constexpr char kInputSize[] = "input_size";

int Main(int argc, char* argv[]) {
  // Command Line Flags.
  std::string model_file_path;
  std::string delegates;
  std::string output_dir;
  int num_threads = 4;
  int min_query_count = 50000, min_duration = 10000;
  int sample_size = 1;
  int input_size = 1;
  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(kModelFileFlag, &model_file_path,
                               "Path to test tflite model file."),
      tflite::Flag::CreateFlag(
          kInterpreterThreadsFlag, &num_threads,
          "Number of interpreter threads to use for inference."),
      tflite::Flag::CreateFlag(
          kDelegateFlag, &delegates,
          "Delegates to use for inference, if available. "
          "Can be one or multiple values of {'nnapi', 'gpu'}."
          "Ex. --delegate=nnapi,gpu"),
      tflite::Flag::CreateFlag(kOutputDirFlag, &output_dir,
                               "The output directory of mlperf."),
      tflite::Flag::CreateFlag(kMinCountFlag, &min_query_count,
                               "The test will guarantee to run at least this "
                               "number of samples in performance mode."),
      tflite::Flag::CreateFlag(
          kMinDurationFlag, &min_duration,
          "The test will guarantee to run at least this "
          "duration in performance mode. The duration is in ms."),
      tflite::Flag::CreateFlag(kSampleSize, &sample_size,
                               "Size of dummy samples."),
      tflite::Flag::CreateFlag(kInputSize, &input_size, "Size of input."),
  };
  tflite::Flags::Parse(&argc, const_cast<const char**>(argv), flag_list);

  // Running mlperf.
  TfliteMlperfDriver driver(
      model_file_path, num_threads, delegates, 1, 1,
      std::unique_ptr<Dataset>(new DummyDataset(sample_size, input_size)));
  driver.StartMLPerfTest("PerformanceOnly", min_query_count, min_duration,
                         output_dir);

  return 0;
}

}  // namespace mlperf
}  // namespace tflite

int main(int argc, char* argv[]) { return tflite::mlperf::Main(argc, argv); }
