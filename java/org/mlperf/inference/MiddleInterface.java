/* Copyright 2020 The MLPerf Authors. All Rights Reserved.

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

import android.content.SharedPreferences;
import android.os.HandlerThread;
import android.util.Base64;
import android.util.Log;
import androidx.preference.PreferenceManager;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import org.mlperf.proto.BackendSetting;
import org.mlperf.proto.BenchmarkSetting;
import org.mlperf.proto.MLPerfConfig;
import org.mlperf.proto.ModelConfig;
import org.mlperf.proto.Setting;
import org.mlperf.proto.TaskConfig;

/** The Interface for UI to interact with middle end. */
final class MiddleInterface implements AutoCloseable, RunMLPerfWorker.Callback {
  public static final String TAG = "MiddleInterface";
  public static final String BACKEND_SETTINGS_KEY_POSFIX = "_backend_settings";
  public static final float SUMMARY_SCORE_MAX = 1000;
  private static final HashMap<String, Integer> defaultSettingMap =
      new HashMap<String, Integer>() {
        {
          put("tflite", R.raw.tflite_setting_pb);
          put("dummy_backend", R.raw.dummy_setting);
        }
      };

  // Non-static variables.
  private String backendName;
  private MLPerfConfig mlperfTasks;
  private SharedPreferences sharedPref;
  private HandlerThread workerThread;
  private RunMLPerfWorker workerHandler;
  private HashMap<String, Setting> settingMap = null;
  private ProgressData progressData;
  private Callback callback;

  public MiddleInterface(String backend, Callback callback) {
    if (!defaultSettingMap.containsKey(backend)) {
      Log.e(TAG, "Backend name not valid");
    }
    backendName = backend;
    this.callback = callback;
    mlperfTasks = MLPerfTasks.getConfig();
    sharedPref = PreferenceManager.getDefaultSharedPreferences(App.getContext());
  }

  // Get the list of benchmarks.
  public ArrayList<Benchmark> getBenchmarks() {
    ArrayList<Benchmark> benchmarks = new ArrayList<>();
    for (TaskConfig task : mlperfTasks.getTaskList()) {
      for (ModelConfig model : task.getModelList()) {
        benchmarks.add(new Benchmark(model));
      }
    }
    return benchmarks;
  }

  // Run all benchmarks with their current settings.
  public void runBenchmarks() {
    if (workerThread == null) {
      workerThread = new HandlerThread("MLPerf.Worker");
      workerThread.start();
      workerHandler = new RunMLPerfWorker(backendName, workerThread.getLooper(), this);
    }

    ArrayList<Benchmark> benchmarks = getBenchmarks();
    progressData = new ProgressData();
    progressData.numBenchmarks = benchmarks.size();
    for (Benchmark bm : benchmarks) {
      String logDir = App.getContext().getExternalFilesDir("log/" + bm.getId()).getAbsolutePath();
      workerHandler.scheduleBenchmark(bm.getId(), logDir);
    }
  }

  // It is not possible to stop the loadgen. So just cancel all waiting jobs.
  public void abortBenchmarks() {
    if (workerThread != null) {
      workerThread.quit();
      workerThread = null;
    }
    // Update the number of benchmarks got run.
    progressData.numBenchmarks = progressData.numStarted;
  }

  public BackendSetting getSettings() throws Exception {
    String backendPref = sharedPref.getString(getBackendKey(), null);
    if (backendPref != null) {
      return BackendSetting.parseFrom(Base64.decode(backendPref, Base64.DEFAULT));
    }

    // If setting not found in the SharedPreference, get the default settings and
    // store it in the SharedPreference.
    InputStream inputStream =
        App.getContext().getResources().openRawResource(defaultSettingMap.get(backendName));
    BackendSetting settings = BackendSetting.parseFrom(inputStream);
    setSetting(settings);
    return settings;
  }

  // Setting will be stored in the SharedPreference. It will be passed to the real backend when
  // creating a C++ backend object to run the benchmark.
  public void setSetting(BackendSetting settings) {
    SharedPreferences.Editor preferencesEditor = sharedPref.edit();
    String settingData = Base64.encodeToString(settings.toByteArray(), Base64.DEFAULT);
    preferencesEditor.putString(getBackendKey(), settingData);
    preferencesEditor.commit();
  }

  // Get a single setting in the common settings section by the setting id.
  public Setting getCommonSetting(String settingId) throws Exception {
    return getSettingMap().get(settingId);
  }

  // Get a single setting in the benchmark settings section.
  public Setting getBenchmarkSetting(String benchmarkId, String settingId) throws Exception {
    return getSettingMap().get(benchmarkId + settingId);
  }

  // run an arbitrary diagnostic command, return arbitrary multi-line text, used by diagnostic
  // screen
  public String runDiagnostic(String command) {
    return "runDiagnostic not yet implemented";
  }

  // get the url to open if user elects to share scores
  public String getShareUrl() {
    return "getShareUrl not yet implemented";
  }

  private HashMap<String, Setting> getSettingMap() throws Exception {
    if (settingMap == null) {
      settingMap = new HashMap<>();
      BackendSetting settings = getSettings();
      for (Setting s : settings.getCommonSettingList()) {
        settingMap.put(s.getId(), s);
      }
      for (BenchmarkSetting bm : settings.getBenchmarkSettingList()) {
        for (Setting s : bm.getSettingList()) {
          settingMap.put(bm.getBenchmarkId() + s.getId(), s);
        }
      }
    }
    return settingMap;
  }

  @Override
  public void close() {
    if (workerThread != null) {
      workerThread.quit();
    }
  }

  @Override
  public void onBenchmarkStarted(String benchmarkId) {
    synchronized (progressData) {
      progressData.numStarted++;
    }
  }

  @Override
  public void onBenchmarkFinished(ResultHolder result) {
    if (callback == null) {
      return;
    }

    // Update the progress.
    synchronized (progressData) {
      progressData.numFinished++;
      progressData.results.add(result);

      callback.onProgressUpdate(progressData.getProgress());
      callback.onbenchmarkFinished(result);
      if (progressData.numFinished == progressData.numBenchmarks) {
        // TODO: Calculate the summary score.
        progressData.summaryScore = 2500;
        callback.onAllBenchmarksFinished(progressData.summaryScore, progressData.results);
      }
    }
  }

  private String getBackendKey() {
    return backendName + BACKEND_SETTINGS_KEY_POSFIX;
  }

  /** Callback with progres update. */
  public interface Callback {
    // Notify a change in the progress.
    public void onProgressUpdate(int percent);

    // Notify that a benchmark is done.
    public void onbenchmarkFinished(ResultHolder result);

    // Notify when all
    public void onAllBenchmarksFinished(float summaryScore, ArrayList<ResultHolder> results);
  }

  // A class to bind all variable related to progress update.
  private class ProgressData {
    // Total number of benchmarks.
    public int numBenchmarks;
    // Number of benchmarks started.
    public int numStarted;
    // Number of benchmarks finished.
    public int numFinished;
    // Summary score will be calculated if all benchmarks finished.
    public float summaryScore;
    // Results of finished benchmarks.
    public ArrayList<ResultHolder> results;

    public ProgressData() {
      numBenchmarks = 0;
      numStarted = 0;
      numFinished = 0;
      summaryScore = 0;
      results = new ArrayList<>();
    }

    public int getProgress() {
      return (100 * numFinished) / numBenchmarks;
    }
  }

  static {
    NativeEvaluation.init();
  }
}
