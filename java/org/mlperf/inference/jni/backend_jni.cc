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

#include "cpp/backends/tflite.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

JNIEXPORT jlong JNICALL Java_org_mlperf_inference_MLPerfDriverWrapper_tflite(
    JNIEnv* env, jclass clazz, jstring jmodel_file_path, jint num_threads,
    jstring jdelegate, jint num_inputs, jint num_outputs) {
  // Convert parameters to C++.
  std::string model_file_path =
      env->GetStringUTFChars(jmodel_file_path, nullptr);
  std::string delegate = env->GetStringUTFChars(jdelegate, nullptr);

  // Create a new TfliteBackend object.
  std::unique_ptr<mlperf::mobile::TfliteBackend> backend_ptr(
      new mlperf::mobile::TfliteBackend(model_file_path, num_threads, delegate,
                                        num_inputs, num_outputs));
  return reinterpret_cast<jlong>(backend_ptr.release());
}

JNIEXPORT void JNICALL
Java_org_mlperf_inference_MLPerfDriverWrapper_nativeDeleteBackend(
    JNIEnv* env, jclass clazz, jlong handle) {
  if (handle != 0) {
    delete reinterpret_cast<mlperf::mobile::Backend*>(handle);
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
