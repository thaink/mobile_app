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

#include "cpp/tasks/coco_object_detection/coco.h"
#include "cpp/tasks/dummy_dataset/dummy_dataset.h"
#include "cpp/tasks/imagenet_classification/imagenet.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

JNIEXPORT jlong JNICALL Java_org_mlperf_inference_MLPerfDriverWrapper_imagenet(
    JNIEnv* env, jclass clazz, jstring jimage_dir, jint offset,
    jboolean jis_raw_images) {
  // Convert parameters to C++.
  std::string image_dir = env->GetStringUTFChars(jimage_dir, nullptr);
  bool is_raw_images = (jis_raw_images == JNI_TRUE);

  // Create a new Imagenet object.
  std::unique_ptr<tflite::mlperf::Imagenet> imagenet_ptr(
      new tflite::mlperf::Imagenet(image_dir, offset, is_raw_images));
  return reinterpret_cast<jlong>(imagenet_ptr.release());
}

JNIEXPORT jlong JNICALL Java_org_mlperf_inference_MLPerfDriverWrapper_coco(
    JNIEnv* env, jclass clazz, jstring jimage_dir, jint offset,
    jboolean jis_raw_images, jstring groundtruth_file) {
  // Convert parameters to C++.
  std::string image_dir = env->GetStringUTFChars(jimage_dir, nullptr);
  std::string gt_file = env->GetStringUTFChars(groundtruth_file, nullptr);
  bool is_raw_images = (jis_raw_images == JNI_TRUE);

  // Create a new Coco object.
  std::unique_ptr<tflite::mlperf::Coco> coco_ptr(
      new tflite::mlperf::Coco(image_dir, offset, is_raw_images, gt_file));
  return reinterpret_cast<jlong>(coco_ptr.release());
}

JNIEXPORT jlong JNICALL
Java_org_mlperf_inference_MLPerfDriverWrapper_dummyDataset(JNIEnv* env,
                                                           jclass clazz,
                                                           jint num_samples,
                                                           jint input_size) {
  // Create a new DummyDataset object.
  std::unique_ptr<tflite::mlperf::DummyDataset> dummy_dataset_ptr(
      new tflite::mlperf::DummyDataset(num_samples, input_size));
  return reinterpret_cast<jlong>(dummy_dataset_ptr.release());
}

JNIEXPORT void JNICALL
Java_org_mlperf_inference_MLPerfDriverWrapper_nativeDeleteDataset(
    JNIEnv* env, jclass clazz, jlong handle) {
  if (handle != 0) {
    delete reinterpret_cast<tflite::mlperf::Dataset*>(handle);
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
