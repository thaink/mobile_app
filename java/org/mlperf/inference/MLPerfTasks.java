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

import android.util.Log;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import org.mlperf.proto.MLPerfConfig;
import org.mlperf.proto.ModelConfig;
import org.mlperf.proto.TaskConfig;

/** This class reads the tasks.pbtxt and provides quick inference to its values. */
final class MLPerfTasks {
  private static final String TAG = "MLPerfTasks";
  private static final String ZIP = ".zip";
  private static MLPerfConfig mlperfTasks;
  private static String localDir;
  // Map a benchmark id to its TaskConfig.
  private static HashMap<String, TaskConfig> taskConfigMap;
  // Map a benchmark id to its ModelConfig.
  private static HashMap<String, ModelConfig> modelConfigMap;

  // Make this class not instantiable.
  private MLPerfTasks() {}

  public static MLPerfConfig getConfig() {
    if (mlperfTasks == null) {
      localDir = App.getContext().getExternalFilesDir("cache").getAbsolutePath();
      try {
        InputStream inputStream = App.getContext().getResources().openRawResource(R.raw.tasks_pb);
        mlperfTasks = MLPerfConfig.parseFrom(inputStream);
        taskConfigMap = new HashMap<>();
        modelConfigMap = new HashMap<>();
        for (TaskConfig task : mlperfTasks.getTaskList()) {
          for (ModelConfig model : task.getModelList()) {
            taskConfigMap.put(model.getId(), task);
            modelConfigMap.put(model.getId(), model);
          }
        }
        inputStream.close();
      } catch (IOException e) {
        Log.e(TAG, "Unable to read config proto file");
      }
    }
    return mlperfTasks;
  }

  public static TaskConfig getTaskConfig(String benchmarkId) {
    if (taskConfigMap == null) {
      getConfig();
    }
    return taskConfigMap.get(benchmarkId);
  }

  public static ModelConfig getModelConfig(String benchmarkId) {
    if (modelConfigMap == null) {
      getConfig();
    }
    return modelConfigMap.get(benchmarkId);
  }

  public static boolean isZipFile(String path) {
    return path.endsWith(ZIP);
  }

  public static String getLocalPath(String path) {
    String filename = new File(path).getName();
    if (isZipFile(filename)) {
      filename = filename.substring(0, filename.length() - ZIP.length());
    }
    return localDir + "/cache/" + filename;
  }
}
