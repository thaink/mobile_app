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
#include <jni.h>

#include <memory>
#include <string>

#include "cpp/dataset.h"
#include "cpp/mlperf_driver.h"
#include "tensorflow/lite/java/src/main/native/jni_utils.h"

std::unique_ptr<tflite::mlperf::Dataset> convertLongToDataset(JNIEnv* env,
                                                              jlong handle) {
  if (handle == 0) {
    ::tflite::jni::ThrowException(env, kIllegalArgumentException,
                                  "Internal error: Invalid handle to Dataset.");
    return nullptr;
  }
  return std::unique_ptr<tflite::mlperf::Dataset>(
      reinterpret_cast<tflite::mlperf::Dataset*>(handle));
}

tflite::mlperf::TfliteMlperfDriver* convertLongToMlperfDriver(JNIEnv* env,
                                                              jlong handle) {
  if (handle == 0) {
    ::tflite::jni::ThrowException(
        env, kIllegalArgumentException,
        "Internal error: Invalid handle to TfliteMlperfDriver.");
    return nullptr;
  }
  return reinterpret_cast<tflite::mlperf::TfliteMlperfDriver*>(handle);
}

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

JNIEXPORT jlong JNICALL
Java_org_mlperf_inference_MLPerfDriverWrapper_nativeInit(
    JNIEnv* env, jclass clazz, jstring jmodel_file_path, jlong dataset_handle,
    jint num_thread, jstring jdelegate, jint num_input, jint num_output) {
  // Convert parameters to C++.
  std::string model_file_path =
      env->GetStringUTFChars(jmodel_file_path, nullptr);
  std::string delegate = env->GetStringUTFChars(jdelegate, nullptr);

  // create a new TfliteMlperfDriver
  std::unique_ptr<tflite::mlperf::TfliteMlperfDriver> driver_ptr(
      new tflite::mlperf::TfliteMlperfDriver(
          model_file_path, num_thread, delegate, num_input, num_output,
          convertLongToDataset(env, dataset_handle)));
  return reinterpret_cast<jlong>(driver_ptr.release());
}

JNIEXPORT void JNICALL Java_org_mlperf_inference_MLPerfDriverWrapper_nativeRun(
    JNIEnv* env, jclass clazz, jlong driver_handle, jstring jmode,
    jint min_query_count, jint min_duration, jstring joutput_dir) {
  // Convert parameters to C++.
  std::string mode = env->GetStringUTFChars(jmode, nullptr);
  std::string output_dir = env->GetStringUTFChars(joutput_dir, nullptr);
  // Start the test.
  convertLongToMlperfDriver(env, driver_handle)
      ->StartMLPerfTest(mode, min_query_count, min_duration, output_dir);
}

JNIEXPORT jstring JNICALL
Java_org_mlperf_inference_MLPerfDriverWrapper_nativeGetLatency(
    JNIEnv* env, jclass clazz, jlong helper_handle) {
  std::string latency =
      convertLongToMlperfDriver(env, helper_handle)->ComputeLatencyString();
  return env->NewStringUTF(latency.c_str());
}

JNIEXPORT jstring JNICALL
Java_org_mlperf_inference_MLPerfDriverWrapper_nativeGetAccuracy(
    JNIEnv* env, jclass clazz, jlong helper_handle, jstring jgt_file) {
  std::string gt_file = env->GetStringUTFChars(jgt_file, nullptr);
  std::string accuracy = convertLongToMlperfDriver(env, helper_handle)
                             ->ComputeAccuracyString(gt_file);
  return env->NewStringUTF(accuracy.c_str());
}

JNIEXPORT void JNICALL
Java_org_mlperf_inference_MLPerfDriverWrapper_nativeDelete(
    JNIEnv* env, jclass clazz, jlong driver_handle) {
  if (driver_handle != 0) {
    delete convertLongToMlperfDriver(env, driver_handle);
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
