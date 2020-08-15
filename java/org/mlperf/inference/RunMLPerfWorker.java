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

import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.Log;
import androidx.annotation.NonNull;
import java.io.File;
import org.mlperf.proto.DatasetConfig;
import org.mlperf.proto.ModelConfig;
import org.mlperf.proto.Setting;
import org.mlperf.proto.TaskConfig;

/**
 * Worker utility to run MLPerf.
 *
 * <p>RunMLPerfWorker is designed to run a single model to avoid restarting a big work if
 * terminated.
 */
public final class RunMLPerfWorker implements Handler.Callback {
  public static final int MSG_RUN = 1;
  public static final String TAG = "RunMLPerfWorker";

  private final Handler handler;
  private final String backendName;
  private final Callback callback;

  public RunMLPerfWorker(String backend, @NonNull Looper looper, @NonNull Callback callback) {
    handler = new Handler(looper, this);
    backendName = backend;
    this.callback = callback;
  }

  @Override
  public boolean handleMessage(Message msg) {
    // Gets the data.
    WorkerData data = (WorkerData) msg.obj;
    Log.i(TAG, "handleMessage() " + data.benchmarkId);
    callback.onBenchmarkStarted(data.benchmarkId);

    // Runs the model.
    String mode = "SubmissionRun";
    TaskConfig taskConfig = MLPerfTasks.getTaskConfig(data.benchmarkId);
    ModelConfig modelConfig = MLPerfTasks.getModelConfig(data.benchmarkId);
    DatasetConfig dataset = taskConfig.getDataset();
    String modelName = modelConfig.getName();
    String runtime;

    boolean useDummyDataSet =
        !dataset.getPath().contains("@assets/")
            && !new File(MLPerfTasks.getLocalPath(dataset.getPath())).isDirectory();
    Log.i(TAG, "Running inference for \"" + modelName + "\"...");
    Log.i(TAG, " - backend: " + backendName);

    try {
      MLPerfDriverWrapper.Builder builder = new MLPerfDriverWrapper.Builder();
      MiddleInterface middleInterface = new MiddleInterface(backendName, null);

      // Set the backend.
      if (backendName.equals("tflite")) {
        Setting.Value numThreads = middleInterface.getCommonSetting("num_threads").getValue();
        Setting.Value accelerator =
            middleInterface.getBenchmarkSetting(modelConfig.getId(), "accelerator").getValue();
        runtime = accelerator.getName();
        if (runtime == "CPU") {
          runtime += "(" + numThreads.getName() + ")";
        }
        builder.useTfliteBackend(
            MLPerfTasks.getLocalPath(modelConfig.getSrc()),
            Integer.parseInt(numThreads.getValue()),
            accelerator.getValue());
      } else if (backendName.equals("dummy_backend")) {
        runtime = "CPU";
        builder.useDummyBackend(MLPerfTasks.getLocalPath(modelConfig.getSrc()));
      } else {
        Log.e(TAG, "The provided backend type is not supported");
        return false;
      }

      // Set the dataset.
      if (useDummyDataSet) {
        builder.useDummy(dataset.getType());
        mode = "PerformanceOnly";
      } else {
        switch (dataset.getType()) {
          case IMAGENET:
            builder.useImagenet(
                MLPerfTasks.getLocalPath(dataset.getPath()),
                MLPerfTasks.getLocalPath(dataset.getGroundtruthSrc()),
                modelConfig.getOffset(),
                /*imageWidth=*/ 224,
                /*imageHeight=*/ 224,
                modelConfig.getScenario());
            break;
          case COCO:
            builder.useCoco(
                MLPerfTasks.getLocalPath(dataset.getPath()),
                MLPerfTasks.getLocalPath(dataset.getGroundtruthSrc()),
                modelConfig.getOffset(),
                /*numClasses=*/ 91,
                /*imageWidth=*/ 300,
                /*imageHeight=*/ 300);
            break;
          case SQUAD:
            builder.useSquad(
                MLPerfTasks.getLocalPath(dataset.getPath()),
                MLPerfTasks.getLocalPath(dataset.getGroundtruthSrc()));
            break;
          case ADE20K:
            builder.useAde20k(
                // The current dataset don't have ground truth images.
                MLPerfTasks.getLocalPath(dataset.getPath()),
                dataset.getGroundtruthSrc(),
                /*numClasses=*/ 31,
                /*imageWidth=*/ 512,
                /*imageHeight=*/ 512);
            break;
        }
      }

      MLPerfDriverWrapper driverWrapper = builder.build();
      driverWrapper.runMLPerf(
          mode,
          modelConfig.getScenario(),
          taskConfig.getMinQueryCount(),
          taskConfig.getMinDurationMs(),
          data.outputFolder);

      Log.i(TAG, "Finished running \"" + modelName + "\".");
      ResultHolder result = new ResultHolder(data.benchmarkId);
      result.setRuntime(runtime);
      result.setScore(driverWrapper.getLatency());
      result.setAccuracy(driverWrapper.getAccuracy());
      callback.onBenchmarkFinished(result);
    } catch (Exception e) {
      Log.e(TAG, "Running \"" + modelName + "\" failed with error: " + e.getMessage());
      Log.e(TAG, Log.getStackTraceString(e));
      return false;
    }
    return true;
  }

  // Schedule a benchmark by sending a message to handler.
  public void scheduleBenchmark(String benchmarkId, String outputFolder) {
    WorkerData data = new WorkerData(benchmarkId, outputFolder);
    Message msg = handler.obtainMessage(MSG_RUN, data);
    handler.sendMessage(msg);
  }

  /** Defines data for this worker. */
  public static class WorkerData {
    public WorkerData(String benchmarkId, String outputFolder) {
      this.benchmarkId = benchmarkId;
      this.outputFolder = outputFolder;
    }

    protected String benchmarkId;
    protected String outputFolder;
  }

  /** Callback interface to return progress and results. */
  public interface Callback {
    // Notify that a benchmark is being run.
    public void onBenchmarkStarted(String benchmarkId);

    // Notify that a benchmark is finished.
    public void onBenchmarkFinished(ResultHolder result);
  }
}
