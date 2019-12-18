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

#include "cpp/backends/tflite.h"
#include "cpp/datasets/coco.h"
#include "cpp/datasets/dummy_dataset.h"
#include "cpp/datasets/imagenet.h"
#include "cpp/mlperf_driver.h"
#include "cpp/utils.h"
#include "tensorflow/lite/tools/command_line_flags.h"

namespace mlperf {
namespace mobile {
namespace {
// Supported backends.
enum BackendType {
  TFLITE = 0,
};

// Supported datasets.
enum DatasetType {
  DUMMY = 0,
  IMAGENET = 1,
  COCO = 2,
};
}  // namespace

int Main(int argc, char* argv[]) {
  using tflite::Flag;
  using tflite::Flags;
  // Command Line Flags for mlperf.
  std::string mode, output_dir;
  int min_query_count = 100, min_duration = 100;
  std::vector<Flag> flag_list = {
      Flag::CreateFlag("mode", &mode,
                       "Required. Mode is one among PerformanceOnly, "
                       "AccuracyOnly, SubmissionRun."),
      Flag::CreateFlag("min_query_count", &min_query_count,
                       "Required. The test will guarantee to run at least this "
                       "number of samples in performance mode."),
      Flag::CreateFlag("min_duration", &min_duration,
                       "Required. The test will guarantee to run at least this "
                       "duration in performance mode. The duration is in ms."),
      Flag::CreateFlag("output_dir", &output_dir,
                       "Required. The output directory of mlperf."),
  };

  // Command Line Flags for backend.
  BackendType backend_type = TFLITE;
  std::unique_ptr<Backend> backend;
#ifdef BACKEND
  backend_type = BACKEND;
#endif
  switch (backend_type) {
    case TFLITE: {
      LOG(INFO) << "Using TFLite backend";
      std::string model_file_path;
      int num_threads = 4, num_inputs = 1, num_outputs = 1;
      std::string delegate;
      flag_list.push_back(
          Flag::CreateFlag("model_file", &model_file_path,
                           "Required. Path to test tflite model file."));
      flag_list.push_back(Flag::CreateFlag(
          "num_threads", &num_threads,
          "Number of interpreter threads to use for inference."));
      flag_list.push_back(
          Flag::CreateFlag("delegate", &delegate,
                           "Delegate to use for inference, if available. "
                           "Can be one value of {'nnapi', 'gpu', 'none'}."));
      flag_list.push_back(
          Flag::CreateFlag("num_inputs", &num_inputs,
                           "Number of inputs required by the model."));
      flag_list.push_back(
          Flag::CreateFlag("num_outputs", &num_outputs,
                           "Number of outputs produced by the model."));
      if (Flags::Parse(&argc, const_cast<const char**>(argv), flag_list)) {
        backend.reset(new TfliteBackend(model_file_path, num_threads, delegate,
                                        num_inputs, num_outputs));
      }
    } break;
  }

  // Command Line Flags for dataset.
  DatasetType dataset_type = DUMMY;
  std::unique_ptr<Dataset> dataset;
#ifdef DATASET
  dataset_type = DATASET;
#endif
  switch (dataset_type) {
    case IMAGENET: {
      LOG(INFO) << "Using Imagenet dataset";
      std::string images_directory, groundtruth_file;
      int offset = 1, image_width, image_height;
      flag_list.push_back(Flag::CreateFlag(
          "images_directory", &images_directory,
          "Required. Path to ground truth images. These will be evaluated in "
          "alphabetical order of filename"));
      flag_list.push_back(Flag::CreateFlag(
          "offset", &offset,
          "The offset of the classification model. Some models with an "
          "additional background class have offset=1 when the background class "
          "has index=0."));
      flag_list.push_back(Flag::CreateFlag(
          "groundtruth_file", &groundtruth_file,
          "Required. Path to the imagenet ground truth file."));
      flag_list.push_back(Flag::CreateFlag(
          "image_width", &image_width,
          "Required. The required width of the processed image."));
      flag_list.push_back(Flag::CreateFlag(
          "image_height", &image_height,
          "Required. The required height of the processed image."));
      if (Flags::Parse(&argc, const_cast<const char**>(argv), flag_list) &&
          backend) {
        dataset.reset(new Imagenet(backend->GetInputFormat(),
                                   backend->GetOutputFormat(), images_directory,
                                   groundtruth_file, offset, image_width,
                                   image_height));
      }
    } break;
    case DUMMY: {
      LOG(INFO) << "Using Dummy dataset";
      if (backend) {
        dataset.reset(new DummyDataset(backend->GetInputFormat(),
                                       backend->GetOutputFormat()));
      }
    } break;
    case COCO: {
      LOG(INFO) << "Using Coco dataset";
      std::string images_directory, groundtruth_file;
      int offset = 1, num_classes = 91, image_width, image_height;
      flag_list.push_back(Flag::CreateFlag(
          "images_directory", &images_directory,
          "Required. Path to ground truth images. These will be evaluated in "
          "alphabetical order of filename"));
      flag_list.push_back(Flag::CreateFlag(
          "offset", &offset,
          "The offset of the classification model. Some models with an "
          "additional background class have offset=1 when the background class "
          "has index=0."));
      flag_list.push_back(
          Flag::CreateFlag("num_classes", &num_classes,
                           "The number of classes in the model outputs."));
      flag_list.push_back(Flag::CreateFlag(
          "groundtruth_file", &groundtruth_file,
          "Required. Path to the imagenet ground truth file."));
      flag_list.push_back(Flag::CreateFlag(
          "image_width", &image_width,
          "Required. The required width of the processed image."));
      flag_list.push_back(Flag::CreateFlag(
          "image_height", &image_height,
          "Required. The required height of the processed image."));
      if (Flags::Parse(&argc, const_cast<const char**>(argv), flag_list) &&
          backend) {
        dataset.reset(new Coco(backend->GetInputFormat(),
                               backend->GetOutputFormat(), images_directory,
                               groundtruth_file, offset, num_classes,
                               image_width, image_height));
      }
    } break;
  }

  // Show usage if needed.
  if (!backend || !dataset) {
    LOG(FATAL) << Flags::Usage(argv[0], flag_list);
    return 1;
  }
  // If using the dummy dataset, only run the performance mode.
  if (dataset_type == DUMMY) {
    mode = "PerformanceOnly";
  }

  // Running mlperf.
  MlperfDriver driver(std::move(dataset), std::move(backend));
  driver.RunMLPerfTest(mode, min_query_count, min_duration, output_dir);
  LOG(INFO) << "90 percentile latency: " << driver.ComputeLatencyString();
  LOG(INFO) << "Accuracy: " << driver.ComputeAccuracyString();
  return 0;
}

}  // namespace mobile
}  // namespace mlperf

int main(int argc, char* argv[]) { return mlperf::mobile::Main(argc, argv); }
