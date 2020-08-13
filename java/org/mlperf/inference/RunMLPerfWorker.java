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
import android.os.Messenger;
import android.os.RemoteException;
import android.util.Log;
import androidx.annotation.NonNull;
import java.io.File;
import java.util.IdentityHashMap;
import java.util.Map;
import org.mlperf.proto.DatasetConfig;
import org.mlperf.proto.ModelConfig;
import org.mlperf.proto.TaskConfig;

/**
 * Worker utility to run MLPerf.
 *
 * <p>RunMLPerfWorker is designed to run a single model to avoid restarting a big work if
 * terminated.
 */
public final class RunMLPerfWorker implements Handler.Callback {
  public static final int MSG_RUN = 1;
  public static final int REPLY_UPDATE = 1;
  public static final int REPLY_COMPLETE = 2;
  public static final int REPLY_CANCEL = 3;
  public static final int REPLY_ERROR = 4;
  public static final String TAG = "RunMLPerfWorker";

  private final IdentityHashMap<Message, String> waitingMessages;
  private final Handler handler;

  public RunMLPerfWorker(@NonNull Looper looper) {
    waitingMessages = new IdentityHashMap<>();
    handler = new Handler(looper, this);
  }

  @Override
  public boolean handleMessage(Message msg) {
    waitingMessages.remove(msg);
    // Gets the data.
    WorkerData data = (WorkerData) msg.obj;
    Messenger messenger = msg.replyTo;
    Log.d(TAG, "handleMessage() " + data.benchmarkId);

    // Runs the model.
    String mode = "SubmissionRun";
    TaskConfig taskConfig = MLPerfTasks.getTaskConfig(data.benchmarkId);
    ModelConfig modelConfig = MLPerfTasks.getModelConfig(data.benchmarkId);
    DatasetConfig dataset = taskConfig.getDataset();
    boolean useDummyDataSet =
        !dataset.getPath().contains("@assets/")
            && !new File(MLPerfTasks.getLocalPath(dataset.getPath())).isDirectory();
    String modelName = modelConfig.getName();
    replyWithUpdateMessage(
        messenger, "Running inference for \"" + modelName + "\"...", REPLY_UPDATE);
    replyWithUpdateMessage(messenger, " - backend: " + data.backend, REPLY_UPDATE);
    try {
      MLPerfDriverWrapper.Builder builder = new MLPerfDriverWrapper.Builder();
      BackendInterface backendInterface = new BackendInterface(data.backend);
      if (data.backend.equals("tflite")) {
        int numThreads =
            Integer.parseInt(backendInterface.getCommonSetting("num_threads").getValue());
        String accelerator =
            backendInterface.getBenchmarkSetting(modelConfig.getId(), "accelerator").getValue();
        builder.useTfliteBackend(
            MLPerfTasks.getLocalPath(modelConfig.getSrc()), numThreads, accelerator);
      } else if (data.backend.equals("dummy_backend")) {
        builder.useDummyBackend(MLPerfTasks.getLocalPath(modelConfig.getSrc()));
      } else {
        replyWithUpdateMessage(
            messenger, "The provided backend type is not supported", REPLY_ERROR);
        return false;
      }
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
      replyWithUpdateMessage(messenger, "Finished running \"" + modelName + "\".", REPLY_UPDATE);
      replyWithCompleteMessage(
          messenger, modelName, driverWrapper.getLatency(), driverWrapper.getAccuracy());
    } catch (Exception e) {
      replyWithUpdateMessage(
          messenger,
          "Running inference for \"" + modelName + "\" failed with error: " + e.getMessage(),
          REPLY_ERROR);
      return false;
    }
    return true;
  }

  // Same as Handler.sendMessage but keeping track of the message pool.
  public boolean sendMessage(Message msg) {
    WorkerData data = (WorkerData) msg.obj;
    String modelName = MLPerfTasks.getModelConfig(data.benchmarkId).getName();
    waitingMessages.put(msg, modelName);
    return handler.sendMessage(msg);
  }

  // Gets a new message for the handler.
  public Message obtainMessage(int what, Object obj) {
    return handler.obtainMessage(what, obj);
  }

  // Clears the message pool.
  public void removeMessages() {
    for (Map.Entry<Message, String> entry : waitingMessages.entrySet()) {
      Message reply = Message.obtain();
      reply.what = REPLY_CANCEL;
      reply.obj = "Canceled worker for \"" + entry.getValue() + "\".";
      try {
        entry.getKey().replyTo.send(reply);
      } catch (RemoteException e) {
        Log.e(TAG, "Failed to send message " + e.getMessage());
      }
    }
    waitingMessages.clear();
  }

  private static void replyWithUpdateMessage(Messenger messenger, String update, int type) {
    Message reply = Message.obtain();
    reply.what = type;
    reply.obj = update;
    try {
      messenger.send(reply);
    } catch (RemoteException e) {
      Log.e(TAG, "Failed to send message " + e.getMessage());
    }
  }

  private static void replyWithCompleteMessage(
      Messenger messenger, String model, String latency, String accuracy) {
    Message reply = Message.obtain();
    reply.what = REPLY_COMPLETE;
    ResultHolder result = new ResultHolder(model);
    result.setInferenceLatency(latency);
    result.setAccuracy(accuracy);
    reply.obj = result;
    try {
      messenger.send(reply);
    } catch (RemoteException e) {
      Log.e(TAG, "Failed to send message " + e.getMessage());
    }
  }

  /** Defines data for this worker. */
  public static class WorkerData {
    public WorkerData(String backend, String benchmarkId, String outputFolder) {
      this.backend = backend;
      this.benchmarkId = benchmarkId;
      this.outputFolder = outputFolder;
    }

    protected String backend;
    protected String benchmarkId;
    protected String outputFolder;
  }
}
