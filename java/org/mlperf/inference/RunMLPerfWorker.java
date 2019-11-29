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
import java.io.IOException;
import java.io.InputStream;
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
  private static final int NUM_SAMPLES_IMAGENET_DUMMY = 100;
  private static final int INPUT_SIZE_IMAGENET_DUMMY = 224 * 224 * 3;
  private static final int NUM_SAMPLES_COCO_DUMMY = 100;
  private static final int INPUT_SIZE_COCO_DUMMY = 300 * 300 * 3;

  private final MLPerfConfig mlperfTasks;

  public RunMLPerfWorker(@NonNull Context context, @NonNull WorkerParameters params)
      throws IOException {
    super(context, params);
    InputStream inputStream =
        getApplicationContext().getResources().openRawResource(R.raw.tasks_pb);
    mlperfTasks = MLPerfConfig.parseFrom(inputStream);
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
      MLPerfDriverWrapper.Builder builder =
          new MLPerfDriverWrapper.Builder(
              modelConfig.getPath(), modelConfig.getNumInputs(), modelConfig.getNumOutputs());
      builder.setNumThreads(numThreads).setDelegate(delegate);
      switch (dataset.getType()) {
        case IMAGENET:
          if (useDummyDataset) {
            builder.useDummy(NUM_SAMPLES_IMAGENET_DUMMY, INPUT_SIZE_IMAGENET_DUMMY);
          } else {
            builder.useImagenet(
                dataset.getPath(), modelConfig.getOffset(), dataset.getIsRawImages());
          }
          break;
        case COCO:
          if (useDummyDataset) {
            builder.useDummy(NUM_SAMPLES_COCO_DUMMY, INPUT_SIZE_COCO_DUMMY);
          } else {
            builder.useCoco(
                dataset.getPath(),
                modelConfig.getOffset(),
                dataset.getIsRawImages(),
                dataset.getGroundtruthPath());
          }
          break;
      }
      MLPerfDriverWrapper driverWrapper = builder.build();
      driverWrapper.runMLPerf(
          "SubmissionRun",
          taskConfig.getMinQueryCount(),
          taskConfig.getMinDurationMs(),
          outputFolder);
      infTimeList = driverWrapper.getLatency();
      accuracyList = driverWrapper.getAccuracy(dataset.getGroundtruthPath());
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
