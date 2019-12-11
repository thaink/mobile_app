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
package org.mlperf.inference;

/** Shared constants in the app. */
final class Constants {
  // Worker Data tags for WorkManager.
  public static final String TASK_INDEX = "TASK_INDEX";
  public static final String MODEL_INDEX = "MODEL_INDEX";
  public static final String DELEGATE = "DELEGATE";
  public static final String OUTPUT_DIR = "OUTPUT_DIR";
  public static final String NUM_THREADS = "NUM_THREADS";
  public static final String USE_DUMMY_DATASET = "USE_DUMMY_DATASET";
  public static final String ERROR_MESSAGE = "ERROR_MESSAGE";
  public static final String LATENCY_RESULT = "LATENCY_RESULT";
  public static final String ACCURACY_RESULT = "ACCURACY_RESULT";

  /* Not instantiable. */
  private Constants() {}
}
