# Copyright 2019 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

common_linkopts = tflite_linkopts() + select({
    "//conditions:default": [],
    "@org_tensorflow//tensorflow:android": [
        "-pie",
    ],
})

cc_library(
    name = "coco",
    srcs = [
        "coco.cc",
    ],
    hdrs = [
        "coco.h",
    ],
    copts = tflite_copts(),
    deps = [
        "//cpp:tfl_mlperf",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_protobuf//:protobuf",
        "@org_tensorflow//tensorflow/core:tflite_portable_logging",
        "@org_tensorflow//tensorflow/lite/tools/evaluation:utils",
        "@org_tensorflow//tensorflow/lite/tools/evaluation/proto:evaluation_stages_cc_proto",
        "@org_tensorflow//tensorflow/lite/tools/evaluation/stages:image_preprocessing_stage",
        "@org_tensorflow//tensorflow/lite/tools/evaluation/stages:object_detection_average_precision_stage",
    ],
)

cc_binary(
    name = "mlperf_object_detection",
    srcs = ["main.cc"],
    copts = tflite_copts(),
    linkopts = common_linkopts,
    deps = [
        ":coco",
        "//cpp:tfl_mlperf",
        "//cpp/tasks:utils",
        "@org_mlperf_inference//:loadgen",
        "@org_tensorflow//tensorflow/core:tflite_portable_logging",
        "@org_tensorflow//tensorflow/lite/tools:command_line_flags",
    ],
)
