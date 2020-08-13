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

#include "cpp/backends/dummy_backend.h"
#include "cpp/backends/tflite.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

JNIEXPORT jlong JNICALL Java_org_mlperf_inference_MLPerfDriverWrapper_tflite(
    JNIEnv* env, jclass clazz, jstring jmodel_file_path, jint num_threads,
    jstring jdelegate) {
  // Convert parameters to C++.
  std::string model_file_path =
      env->GetStringUTFChars(jmodel_file_path, nullptr);
  std::string delegate = env->GetStringUTFChars(jdelegate, nullptr);

  // Create a new TfliteBackend object.
  std::unique_ptr<mlperf::mobile::TfliteBackend> backend_ptr(
      new mlperf::mobile::TfliteBackend(model_file_path, num_threads));
  if (backend_ptr->ApplyDelegate(delegate) != 0) {
    env->ThrowNew(env->FindClass("java/lang/Exception"),
                  "failed to apply delegate");
  }
  return reinterpret_cast<jlong>(backend_ptr.release());
}

JNIEXPORT jlong JNICALL
Java_org_mlperf_inference_MLPerfDriverWrapper_dummyBackend(
    JNIEnv* env, jclass clazz, jstring jmodel_file_path) {
  // Convert parameters to C++.
  std::string model_file_path =
      env->GetStringUTFChars(jmodel_file_path, nullptr);

  // Create a new TfliteBackend object.
  std::unique_ptr<mlperf::mobile::DummyBackend> backend_ptr(
      new mlperf::mobile::DummyBackend(model_file_path));
  return reinterpret_cast<jlong>(backend_ptr.release());
}

JNIEXPORT void JNICALL
Java_org_mlperf_inference_MLPerfDriverWrapper_nativeDeleteBackend(
    JNIEnv* env, jclass clazz, jlong handle) {
  if (handle != 0) {
    delete reinterpret_cast<mlperf::mobile::Backend*>(handle);
  }
}

// Get setting as serialized string.
JNIEXPORT jbyteArray JNICALL
Java_org_mlperf_inference_MLPerfDriverWrapper_getBackendSettings(
    JNIEnv* env, jclass clazz, jlong backend_handle) {
  std::string settings;
  reinterpret_cast<mlperf::mobile::Backend*>(backend_handle)
      ->GetSettings()
      .SerializeToString(&settings);

  // Copy to jbyteArray.
  jbyteArray array = env->NewByteArray(settings.size());
  env->SetByteArrayRegion(array, 0, settings.size(),
                          reinterpret_cast<const jbyte*>(settings.c_str()));
  return array;
}

// Get setting as serialized string.
JNIEXPORT void JNICALL
Java_org_mlperf_inference_MLPerfDriverWrapper_setBackendSettings(
    JNIEnv* env, jclass clazz, jlong backend_handle, jbyteArray jsettings) {
  int len = env->GetArrayLength(jsettings);
  char* buf = new char[len];
  env->GetByteArrayRegion(jsettings, 0, len, reinterpret_cast<jbyte*>(buf));
  std::string settings_str(buf, len);
  delete[] buf;

  mlperf::mobile::BackendSetting settings;
  settings.ParseFromString(settings_str);
  reinterpret_cast<mlperf::mobile::Backend*>(backend_handle)
      ->SetSettings(settings);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
