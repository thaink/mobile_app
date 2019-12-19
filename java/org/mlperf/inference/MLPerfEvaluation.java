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

import android.animation.ArgbEvaluator;
import android.animation.ObjectAnimator;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.os.AsyncTask;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.Observer;
import androidx.recyclerview.widget.DefaultItemAnimator;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.work.Data;
import androidx.work.ExistingWorkPolicy;
import androidx.work.OneTimeWorkRequest;
import androidx.work.WorkInfo;
import androidx.work.WorkManager;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.UUID;
import org.mlperf.proto.DatasetConfig;
import org.mlperf.proto.MLPerfConfig;
import org.mlperf.proto.ModelConfig;
import org.mlperf.proto.TaskConfig;

/** {@link MLPerfEvaluation} evaluates models on MLPerf benchmark. */
public class MLPerfEvaluation extends AppCompatActivity {
  private static final String TAG = "MLPerfEvaluation";
  private static final String WORKER_NAME = "MLPerfInferenceWorker";
  private static final String PID_TAG = "PID";
  private static final String ASSETS_PREFIX = "@assets/";

  private ProgressCount progressCount;
  private TextView taskResultText;
  private View dividerBar;
  private RecyclerView resultRecyclerView;
  private ResultsAdapter resultAdapter;
  private final ArrayList<ResultHolder> results = new ArrayList<>();
  HashMap<String, Integer> resultMap = new HashMap<>();

  private String interpreterDelegate;
  private int numThreadsPreference;
  private int highLightColor;
  private int backgroundColor;

  private MLPerfConfig mlperfTasks;
  private final HashSet<UUID> workerInQueue = new HashSet<>();
  private boolean modelIsAvailable = false;

  @Override
  public void onCreate(@Nullable Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    // Set up the RecyclerView which shows results.
    resultRecyclerView = findViewById(R.id.results_recycler_view);
    resultRecyclerView.setLayoutManager(new LinearLayoutManager(this));
    resultRecyclerView.setItemAnimator(new ResultItemAnimator());
    resultAdapter = new ResultsAdapter(this, results);
    resultRecyclerView.setAdapter(resultAdapter);
    highLightColor = ContextCompat.getColor(this, R.color.mlperfBlue);
    backgroundColor = ContextCompat.getColor(this, R.color.background);

    // Set up progress bar and log area.
    ProgressBar progressBar = findViewById(R.id.progressBar);
    taskResultText = findViewById(R.id.taskResultText);
    taskResultText.setMovementMethod(new ScrollingMovementMethod());
    dividerBar = findViewById(R.id.divider);

    // Set up menu buttons.
    ImageView playButton = findViewById(R.id.action_play);
    playButton.setOnClickListener(playButtonListener);
    ImageView stopButton = findViewById(R.id.action_stop);
    stopButton.setOnClickListener(stopButtonListener);
    ImageView refreshButton = findViewById(R.id.action_refresh);
    refreshButton.setOnClickListener(refreshButtonListener);
    ImageView settingButton = findViewById(R.id.action_settings);
    settingButton.setOnClickListener(settingButtonListener);

    // Read tasks from proto file.
    try {
      InputStream inputStream =
          getApplicationContext().getResources().openRawResource(R.raw.tasks_pb);
      mlperfTasks = MLPerfConfig.parseFrom(inputStream);
    } catch (IOException e) {
      Log.e(TAG, "failed to read the proto file", e);
      logProgress("Failed to load the proto file.");
      return;
    }

    checkModelIsAvailable();
    progressCount = new ProgressCount(progressBar);
    // When a task was running, onDestroy might not be called when the previous session ended;
    // Instead the process get killed. Cancel all tasks of the previous session in such case.
    int pid = android.os.Process.myPid();
    SharedPreferences sharedPref = PreferenceManager.getDefaultSharedPreferences(this);
    if (pid != sharedPref.getInt(PID_TAG, -1)) {
      WorkManager.getInstance(MLPerfEvaluation.this).cancelUniqueWork(WORKER_NAME);
    }
    SharedPreferences.Editor preferencesEditor = sharedPref.edit();
    preferencesEditor.putInt(PID_TAG, pid);
    preferencesEditor.commit();
  }

  @Override
  public void onResume() {
    super.onResume();
    SharedPreferences sharedPref =
        PreferenceManager.getDefaultSharedPreferences(MLPerfEvaluation.this);
    interpreterDelegate =
        sharedPref.getString(
            getString(R.string.pref_delegate_key), getString(R.string.delegate_nnapi));
    numThreadsPreference =
        Integer.parseInt(
            sharedPref.getString(
                getString(R.string.num_threads_key), getString(R.string.num_threads_default)));
    String logInfoPreference =
        sharedPref.getString(getString(R.string.pref_loginfo_key), getString(R.string.log_short));
    if (logInfoPreference.equals(getString(R.string.log_short))) {
      dividerBar.setVisibility(View.VISIBLE);
      taskResultText.setVisibility(View.VISIBLE);
      taskResultText.setMaxLines(4);
    } else if (logInfoPreference.equals(getString(R.string.log_full))) {
      dividerBar.setVisibility(View.VISIBLE);
      taskResultText.setVisibility(View.VISIBLE);
      taskResultText.setMaxLines(12);
    } else if (logInfoPreference.equals(getString(R.string.log_none))) {
      dividerBar.setVisibility(View.INVISIBLE);
      taskResultText.setVisibility(View.GONE);
    } else {
      dividerBar.setVisibility(View.INVISIBLE);
      taskResultText.setVisibility(View.GONE);
      Log.e(TAG, "Unknown LogInfo perference value: " + logInfoPreference);
    }
  }

  private void logProgress(String msg) {
    taskResultText.append(System.getProperty("line.separator"));
    taskResultText.append(msg);
    Log.i(TAG, "logProgress: " + msg);
  }

  private final View.OnClickListener playButtonListener =
      new View.OnClickListener() {
        @Override
        public void onClick(View v) {
          WorkManager.getInstance(MLPerfEvaluation.this).pruneWork();
          if (checkModelIsAvailable()) {
            for (int taskIdx = 0; taskIdx < mlperfTasks.getTaskCount(); ++taskIdx) {
              for (int modelIdx = 0;
                  modelIdx < mlperfTasks.getTask(taskIdx).getModelCount();
                  ++modelIdx) {
                scheduleInference(taskIdx, modelIdx);
              }
            }
          } else {
            logProgress("models are not available.");
          }
        }
      };

  private final View.OnClickListener stopButtonListener =
      new View.OnClickListener() {
        @Override
        public void onClick(View v) {
          // Multiple MLPerf tasks should not be runned at the same time (currently, it will crash
          // and it is also not good for the performance). So the running task should not be
          // canceled; because in that case, the completion of that task cannot be tracked. Only
          // tasks in queue will be canceled.
          for (UUID workerID : workerInQueue) {
            WorkManager.getInstance(MLPerfEvaluation.this).cancelWorkById(workerID);
          }
        }
      };

  private final View.OnClickListener refreshButtonListener =
      new View.OnClickListener() {
        @Override
        public void onClick(View v) {
          results.clear();
          resultMap.clear();
          resultAdapter.notifyDataSetChanged();
        }
      };

  private final View.OnClickListener settingButtonListener =
      new View.OnClickListener() {
        @Override
        public void onClick(View v) {
          Intent intent = new Intent(MLPerfEvaluation.this, SettingsActivity.class);
          startActivityForResult(intent, 0);
        }
      };

  // Animate to highlight newly changed item in the results.
  private boolean animateBackground(RecyclerView.ViewHolder holder) {
    if (holder == null) {
      return true;
    }
    ObjectAnimator bgAnimator =
        ObjectAnimator.ofObject(
            holder.itemView,
            "backgroundColor",
            new ArgbEvaluator(),
            highLightColor,
            backgroundColor);
    bgAnimator.setDuration(4000);
    bgAnimator.start();
    return true;
  }
  ;

  // The Animator that uses animateBackground.
  private class ResultItemAnimator extends DefaultItemAnimator {
    @Override
    public boolean animateAdd(RecyclerView.ViewHolder holder) {
      return animateBackground(holder);
    }

    @Override
    public boolean animateChange(
        RecyclerView.ViewHolder oldHolder,
        RecyclerView.ViewHolder newHolder,
        int fromX,
        int fromY,
        int toX,
        int toY) {
      return animateBackground(newHolder);
    }
  }

  // Schedule a inference task with WorkManager for the given model.
  private void scheduleInference(int taskIdx, int modelIdx) {
    Log.d(TAG, "scheduleInference " + taskIdx + " , " + modelIdx);
    TaskConfig task = mlperfTasks.getTask(taskIdx);
    final String modelName = task.getModel(modelIdx).getName();
    DatasetConfig dataset = task.getDataset();
    boolean useDummyDataSet = !new File(dataset.getPath()).isDirectory();
    Log.d(TAG, "Checking dataset path " + dataset.getPath() + ": " + !useDummyDataSet);
    if (useDummyDataSet) {
      logProgress("Dataset for \"" + modelName + "\" is not available. Dummy dataset is used.");
    }
    String outputLogDir =
        getApplicationContext().getExternalFilesDir("mlperf/" + modelName).getAbsolutePath();
    Log.i(TAG, "The mlperf log dir for \"" + modelName + "\" is " + outputLogDir + "/");

    Data taskData =
        new Data.Builder()
            .putInt(Constants.TASK_INDEX, taskIdx)
            .putInt(Constants.MODEL_INDEX, modelIdx)
            .putInt(Constants.NUM_THREADS, numThreadsPreference)
            .putString(Constants.DELEGATE, interpreterDelegate)
            .putString(Constants.OUTPUT_DIR, outputLogDir)
            .putBoolean(Constants.USE_DUMMY_DATASET, useDummyDataSet)
            .build();
    OneTimeWorkRequest mlperfWorkRequest =
        new OneTimeWorkRequest.Builder(RunMLPerfWorker.class).setInputData(taskData).build();
    final String runtime = computeRuntimeString(numThreadsPreference, interpreterDelegate);

    // Send and observe the status.
    WorkManager.getInstance(MLPerfEvaluation.this)
        .enqueueUniqueWork(WORKER_NAME, ExistingWorkPolicy.APPEND, mlperfWorkRequest);
    workerInQueue.add(mlperfWorkRequest.getId());
    progressCount.increaseTotal();
    WorkManager.getInstance(MLPerfEvaluation.this)
        .getWorkInfoByIdLiveData(mlperfWorkRequest.getId())
        .observe(
            MLPerfEvaluation.this,
            new Observer<WorkInfo>() {
              @Override
              public void onChanged(@Nullable WorkInfo workInfo) {
                if (workInfo == null) {
                  return;
                }
                switch (workInfo.getState()) {
                  case BLOCKED:
                    logProgress("Worker for \"" + modelName + "\" scheduled.");
                    break;
                  case FAILED:
                    Data failData = workInfo.getOutputData();
                    logProgress(
                        "Running inference for \""
                            + modelName
                            + "\" failed with "
                            + failData.getString(Constants.ERROR_MESSAGE));
                    addNewResult(modelName, runtime, "ERR", "ERR");
                    progressCount.increaseProgress();
                    break;
                  case RUNNING:
                    logProgress("Running inference for \"" + modelName + "\"...");
                    logProgress(" - runtime: " + runtime);
                    workerInQueue.remove(workInfo.getId());
                    break;
                  case CANCELLED:
                    logProgress("Canceled worker for \"" + modelName + "\".");
                    progressCount.decreaseTotal();
                    break;
                  case SUCCEEDED:
                    logProgress("Finished running \"" + modelName + "\".");
                    Data successData = workInfo.getOutputData();
                    String infTime = successData.getString(Constants.LATENCY_RESULT);
                    String accuracy = successData.getString(Constants.ACCURACY_RESULT);
                    addNewResult(modelName, runtime, infTime, accuracy);
                    progressCount.increaseProgress();
                    break;
                  default:
                    break;
                }
              }
            });
  }

  private static class ProgressCount {
    public ProgressCount(ProgressBar progBar) {
      totalWorkCount = 0;
      finishedWorkCount = 0;
      progressBar = progBar;
      updateUI();
    }

    public synchronized void increaseTotal() {
      ++totalWorkCount;
      updateUI();
    }

    public synchronized void decreaseTotal() {
      --totalWorkCount;
      updateUI();
    }

    public synchronized void increaseProgress() {
      ++finishedWorkCount;
      updateUI();
    }

    public void updateUI() {
      if (totalWorkCount == 0) {
        progressBar.setProgress(0);
        return;
      }
      progressBar.setProgress(finishedWorkCount * 100 / totalWorkCount);
    }

    private int totalWorkCount;
    private int finishedWorkCount;
    private final ProgressBar progressBar;
  }

  private void addNewResult(String modelName, String runtime, String infTime, String accuracy) {
    String key = modelName + runtime;
    int resultIdx;
    // If a result of (model, runtime) is already displayed, update it.
    if (resultMap.containsKey(key)) {
      resultIdx = resultMap.get(key);
      results.get(resultIdx).setRuntime(runtime);
      results.get(resultIdx).setInferenceLatency(infTime);
      results.get(resultIdx).setAccuracy(accuracy);
      resultAdapter.notifyItemChanged(resultIdx);
    } else {
      resultIdx = results.size();
      resultMap.put(key, resultIdx);
      results.add(new ResultHolder(modelName));
      results.get(resultIdx).setRuntime(runtime);
      results.get(resultIdx).setInferenceLatency(infTime);
      results.get(resultIdx).setAccuracy(accuracy);
      resultAdapter.notifyItemInserted(resultIdx);
    }

    // If scrolling happened, the animation does not run. So call it explicitly in such case.
    resultRecyclerView.clearOnScrollListeners();
    resultRecyclerView.addOnScrollListener(
        new RecyclerView.OnScrollListener() {
          @Override
          public void onScrolled(RecyclerView recyclerView, int dx, int dy) {
            RecyclerView.ViewHolder viewHolder =
                recyclerView.findViewHolderForAdapterPosition(resultIdx);
            animateBackground(viewHolder);
            recyclerView.clearOnScrollListeners();
          }
        });
    // Scroll to recently changed item.
    resultRecyclerView.smoothScrollToPosition(resultIdx);
  }

  // Build a string to present the runtime setting.
  private static String computeRuntimeString(int numThreads, String delegate) {
    StringBuilder runtimeStr = new StringBuilder();
    if ("none".equalsIgnoreCase(delegate)) {
      runtimeStr.append("CPU, ");
      runtimeStr.append(numThreads);
      runtimeStr.append(" thread");
      if (numThreads > 1) {
        runtimeStr.append("s");
      }
    } else {
      runtimeStr.append(delegate);
    }
    return runtimeStr.toString();
  }

  public Context getActivityContext() {
    return this;
  }

  @Override
  public void onDestroy() {
    super.onDestroy();
    for (UUID workerID : workerInQueue) {
      WorkManager.getInstance(MLPerfEvaluation.this).cancelWorkById(workerID);
    }
    Log.d(TAG, "onDestroy() is called.");
  }

  private boolean checkModelIsAvailable() {
    if (checkSelfPermission(android.Manifest.permission.WRITE_EXTERNAL_STORAGE)
        == PackageManager.PERMISSION_GRANTED) {
      return doCheckModelIsAvailable();
    } else {
      ActivityCompat.requestPermissions(
          this, new String[] {android.Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
      return false;
    }
  }

  @Override
  public void onRequestPermissionsResult(
      int requestCode, String[] permissions, int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    switch (requestCode) {
      case 1:
        if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
          doCheckModelIsAvailable();
        } else {
          logProgress("Warning: You need to grant external storage access to use MLPerf app.");
        }
        break;
      default:
        break;
    }
  }

  private boolean doCheckModelIsAvailable() {
    if (!modelIsAvailable) {
      for (TaskConfig task : mlperfTasks.getTaskList()) {
        for (ModelConfig model : task.getModelList()) {
          if (!new File(model.getPath()).canRead()) {
            new ModelExtractTask(getActivityContext(), mlperfTasks).execute();
            return false;
          }
        }
        DatasetConfig dataset = task.getDataset();
        if (!new File(dataset.getGroundtruthPath()).canRead()) {
          new ModelExtractTask(getActivityContext(), mlperfTasks).execute();
          return false;
        }
      }

      setModelIsAvailable();
    }
    return true;
  }

  private void setModelIsAvailable() {
    modelIsAvailable = true;
    logProgress("Ready. Click the Run button to evaluate.");
  }

  // ModelExtractTask copies or downloads files to their location (external storage) when
  // they're not available.
  private static class ModelExtractTask extends AsyncTask<Void, Void, Void> {

    private final Context context;
    private final MLPerfConfig mlperfTasks;

    public ModelExtractTask(Context context, MLPerfConfig mlperfTasks) {
      this.context = context;
      this.mlperfTasks = mlperfTasks;
    }

    @Override
    protected Void doInBackground(Void... voids) {
      boolean success = true;
      for (TaskConfig task : mlperfTasks.getTaskList()) {
        for (ModelConfig model : task.getModelList()) {
          if (!new File(model.getPath()).canRead()) {
            if (!extractFile(model.getSrc(), model.getPath())) {
              success = false;
            }
          }
        }
        DatasetConfig dataset = task.getDataset();
        if (!new File(dataset.getGroundtruthPath()).canRead()) {
          if (!extractFile(dataset.getGroundtruthSrc(), dataset.getGroundtruthPath())) {
            success = false;
          }
        }
      }
      if (success) {
        Log.d(TAG, "All missing files are extracted.");
      }
      return null;
    }

    private boolean extractFile(String src, String path) {
      File destFile = new File(path);
      Log.d(TAG, "preparing " + destFile.getName());
      destFile.getParentFile().mkdirs();
      try {
        InputStream in;
        if (src.startsWith(ASSETS_PREFIX)) {
          AssetManager assetManager = context.getAssets();
          in = assetManager.open(src.substring(ASSETS_PREFIX.length()));
        } else if (src.startsWith("http://") || src.startsWith("https://")) {
          in = new URL(src).openStream();
        } else {
          Log.e(TAG, "malformed path: " + src);
          return false;
        }
        OutputStream out = new FileOutputStream(path);
        copyFile(in, out);
      } catch (IOException e) {
        Log.e(TAG, "failed to prepare file: " + path, e);
        return false;
      }

      Log.d(TAG, destFile.getName() + " are extracted.");
      return true;
    }

    private static void copyFile(InputStream in, OutputStream out) throws IOException {
      byte[] buffer = new byte[1024];
      int read;
      while ((read = in.read(buffer)) != -1) {
        out.write(buffer, 0, read);
      }
    }
  }
}
