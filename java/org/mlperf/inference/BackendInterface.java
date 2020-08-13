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
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Message;
import android.os.Messenger;
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

/* A class that UI can interact with middle/backend Part. */
final class BackendInterface implements Handler.Callback, AutoCloseable {
  public static final String TAG = "BackendInterface";
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
  private MLPerfConfig mlperfTasks;
  private SharedPreferences sharedPref;
  private HandlerThread workerThread;
  private RunMLPerfWorker workerHandler;
  private Messenger replyMessenger;
  private String backendName;
  private HashMap<String, Setting> settingMap = null;

  public BackendInterface(String backend) {
    if (!defaultSettingMap.containsKey(backend)) {
      Log.e(TAG, "Backend name not valid");
    }
    backendName = backend;
    mlperfTasks = MLPerfTasks.getConfig();
    sharedPref = PreferenceManager.getDefaultSharedPreferences(App.getContext());
  }

  // Get the list of benchmark.
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
      workerHandler = new RunMLPerfWorker(workerThread.getLooper());
      replyMessenger = new Messenger(new Handler(App.getContext().getMainLooper(), this));
    }

    ArrayList<Benchmark> benchmarks = getBenchmarks();
    for (Benchmark bm : benchmarks) {
      String logDir = App.getContext().getExternalFilesDir("log/" + bm.getId()).getAbsolutePath();
      RunMLPerfWorker.WorkerData data =
          new RunMLPerfWorker.WorkerData(backendName, bm.getId(), logDir);
      Message msg = workerHandler.obtainMessage(RunMLPerfWorker.MSG_RUN, data);
      msg.replyTo = replyMessenger;
      workerHandler.sendMessage(msg);
    }
  }

  // It is not possible to stop the current loadgen. So just cancel all waiting jobs.
  public void abortBenchmarks() {
    if (workerThread != null) {
      workerHandler.removeMessages();
      workerThread.quit();
      workerThread = null;
    }
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

  // Get a single setting int the commen settings section by the setting id.
  public Setting getCommonSetting(String settingId) throws Exception {
    return getSettingMap().get(settingId);
  }

  // Get a single setting int the benchmark settings sectio.
  public Setting getBenchmarkSetting(String benchmarkId, String settingId) throws Exception {
    return getSettingMap().get(benchmarkId + settingId);
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
  public boolean handleMessage(Message inputMessage) {
    switch (inputMessage.what) {
      case RunMLPerfWorker.REPLY_UPDATE:
        String update = (String) inputMessage.obj;
        Log.e(TAG, "got update message: " + update);
        break;
      case RunMLPerfWorker.REPLY_COMPLETE:
        Log.e(TAG, "got complete message: ");
        break;
      case RunMLPerfWorker.REPLY_ERROR:
        String error = (String) inputMessage.obj;
        Log.e(TAG, "got error message: " + error);
      default:
        return false;
    }
    return true;
  }

  @Override
  public void close() {
    if (workerThread != null) {
      workerThread.quit();
    }
  }

  private String getBackendKey() {
    return backendName + BACKEND_SETTINGS_KEY_POSFIX;
  }

  static {
    NativeEvaluation.init();
  }
}
