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

import android.content.Context;
import android.util.Log;
import androidx.annotation.NonNull;
import androidx.work.Data;
import androidx.work.Worker;
import androidx.work.WorkerParameters;
import org.mlperf.proto.DatasetConfig;
import org.mlperf.proto.MLPerfConfig;
import org.mlperf.proto.ModelConfig;
import org.mlperf.proto.TaskConfig;

/**
 * Worker utility to run MLPerf.
 *
 * <p>RunMLPerfWorker is designed to run a single model to avoid restarting a big work if
 * terminated.
 */
public final class RunMLPerfWorker extends Worker {
  private static final String TAG = "RunMLPerfWorker";

  private final MLPerfConfig mlperfTasks;

  public RunMLPerfWorker(@NonNull Context context, @NonNull WorkerParameters params) {
    super(context, params);
    mlperfTasks = MLPerfTasks.getConfig(context);
  }

  @Override
  public Result doWork() {
    // Get the data.
    Data inputData = getInputData();
    Log.d(TAG, "doWork() " + inputData);
    int taskIdx = inputData.getInt(Constants.TASK_INDEX, -1);
    int modelIdx = inputData.getInt(Constants.MODEL_INDEX, -1);
    int numThreads = inputData.getInt(Constants.NUM_THREADS, -1);
    String delegate = inputData.getString(Constants.DELEGATE);
    String outputFolder = inputData.getString(Constants.OUTPUT_DIR);
    boolean useDummyDataset = inputData.getBoolean(Constants.USE_DUMMY_DATASET, false);
    // Validate data.
    if (taskIdx < 0 || modelIdx < 0 || numThreads < 0) {
      Data outputData =
          new Data.Builder().putString(Constants.ERROR_MESSAGE, "Received malformed data").build();
      return Result.failure(outputData);
    }
    // Run each model.
    String infTimeList;
    String accuracyList;
    TaskConfig taskConfig = mlperfTasks.getTask(taskIdx);
    ModelConfig modelConfig = taskConfig.getModel(modelIdx);
    DatasetConfig dataset = taskConfig.getDataset();
    try {
      MLPerfDriverWrapper.Builder builder = new MLPerfDriverWrapper.Builder();
      builder.useTfliteBackend(
          modelConfig.getPath(),
          numThreads,
          delegate,
          modelConfig.getNumInputs(),
          modelConfig.getNumOutputs());
      if (useDummyDataset) {
        builder.useDummy();
      } else {
        switch (dataset.getType()) {
          case IMAGENET:
            builder.useImagenet(
                dataset.getPath(),
                dataset.getGroundtruthPath(),
                modelConfig.getOffset(),
                /*imageWidth=*/ 224,
                /*imageHeight=*/ 224);
            break;
          case COCO:
            builder.useCoco(
                dataset.getPath(),
                dataset.getGroundtruthPath(),
                modelConfig.getOffset(),
                /*numClasses=*/ 91,
                /*imageWidth=*/ 300,
                /*imageHeight=*/ 300);
            break;
        }
      }
      MLPerfDriverWrapper driverWrapper = builder.build();
      driverWrapper.runMLPerf(
          "SubmissionRun",
          taskConfig.getMinQueryCount(),
          taskConfig.getMinDurationMs(),
          outputFolder);
      infTimeList = driverWrapper.getLatency();
      accuracyList = driverWrapper.getAccuracy();
    } catch (Exception e) {
      Log.e(TAG, "Failed to run the model: " + e.getMessage());
      Data outputData =
          new Data.Builder().putString(Constants.ERROR_MESSAGE, e.getMessage()).build();
      return Result.failure(outputData);
    }

    // Create the output of the work
    Data outputData =
        new Data.Builder()
            .putString(Constants.LATENCY_RESULT, infTimeList)
            .putString(Constants.ACCURACY_RESULT, accuracyList)
            .build();
    // Return the output
    return Result.success(outputData);
  }
}
