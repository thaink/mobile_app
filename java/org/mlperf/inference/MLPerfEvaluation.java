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
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.ProgressBar;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.preference.PreferenceManager;
import androidx.recyclerview.widget.DefaultItemAnimator;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.ref.WeakReference;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import org.mlperf.proto.DatasetConfig;
import org.mlperf.proto.MLPerfConfig;
import org.mlperf.proto.ModelConfig;
import org.mlperf.proto.TaskConfig;

/** {@link MLPerfEvaluation} evaluates models on MLPerf benchmark. */
public class MLPerfEvaluation extends AppCompatActivity
    implements Handler.Callback, MiddleInterface.Callback {

  private static final String TAG = "MLPerfEvaluation";
  private static final String ASSETS_PREFIX = "@assets/";
  private static final int MSG_PROGRESS = 1;
  private static final int MSG_COMPLETE = 2;

  private RecyclerView resultRecyclerView;
  private ResultsAdapter resultAdapter;
  private ProgressBar progressBar;
  private final ArrayList<ResultHolder> results = new ArrayList<>();
  private final HashMap<String, Integer> resultMap = new HashMap<>();

  private String backend;
  private int highLightColor;
  private int backgroundColor;

  private MLPerfConfig mlperfTasks;
  private Handler handler;

  private boolean modelIsAvailable = false;
  private SharedPreferences sharedPref;
  private MiddleInterface middleInterface;

  @Override
  public void onCreate(@Nullable Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    // Sets up the RecyclerView which shows results.
    resultRecyclerView = findViewById(R.id.results_recycler_view);
    resultRecyclerView.setLayoutManager(new LinearLayoutManager(this));
    resultRecyclerView.setItemAnimator(new ResultItemAnimator());
    resultAdapter = new ResultsAdapter(this, results);
    resultRecyclerView.setAdapter(resultAdapter);
    highLightColor = ContextCompat.getColor(this, R.color.mlperfBlue);
    backgroundColor = ContextCompat.getColor(this, R.color.background);

    // Sets up progress bar.
    progressBar = findViewById(R.id.progressBar);

    // Sets up menu buttons.
    ImageView playButton = findViewById(R.id.action_play);
    playButton.setOnClickListener(this::playButtonListener);
    ImageView stopButton = findViewById(R.id.action_stop);
    stopButton.setOnClickListener(this::stopButtonListener);
    ImageView refreshButton = findViewById(R.id.action_refresh);
    refreshButton.setOnClickListener(this::refreshButtonListener);
    ImageView settingButton = findViewById(R.id.action_settings);
    settingButton.setOnClickListener(this::settingButtonListener);

    // Create a handler to update UI based on callback.
    handler = new Handler(this.getMainLooper(), this);
  }

  @Override
  public void onResume() {
    super.onResume();
    mlperfTasks = MLPerfTasks.getConfig();
    sharedPref = PreferenceManager.getDefaultSharedPreferences(this);

    // Checks if models are available.
    checkModelIsAvailable();

    // Updates the shared preference.
    backend =
        sharedPref.getString(
            getString(R.string.backend_preference_key), getString(R.string.tflite_preference_key));
    middleInterface = new MiddleInterface(backend, MLPerfEvaluation.this);
  }

  @Override
  public boolean handleMessage(Message inputMessage) {
    switch (inputMessage.what) {
      case MSG_PROGRESS:
        int percent = inputMessage.arg1;
        progressBar.setProgress(percent);
        break;
      case MSG_COMPLETE:
        ResultHolder result = (ResultHolder) inputMessage.obj;
        addNewResult(result);
        break;
      default:
        return false;
    }
    return true;
  }

  private void playButtonListener(View v) {
    if (checkModelIsAvailable()) {
      getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
      middleInterface.runBenchmarks();
    } else {
      Log.i(TAG, "Models are not available.");
    }
  }

  private void stopButtonListener(View v) {
    middleInterface.abortBenchmarks();
  }

  private void refreshButtonListener(View v) {
    results.clear();
    resultMap.clear();
    resultAdapter.notifyDataSetChanged();
    // Fix: if an item is updated, the cache of older item is visable after deleting newer one.
    resultRecyclerView.setLayoutManager(new LinearLayoutManager(this));
  }

  private void settingButtonListener(View v) {
    Intent intent = new Intent(MLPerfEvaluation.this, SettingsActivity.class);
    startActivityForResult(intent, 0);
  }

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

  private void addNewResult(ResultHolder result) {
    String key = result.getId();
    int resultIdx;
    // If a result of (model, runtime) is already displayed, update it.
    if (resultMap.containsKey(key)) {
      resultIdx = resultMap.get(key);
      results.set(resultIdx, result);
      resultAdapter.notifyItemChanged(resultIdx);
    } else {
      resultIdx = results.size();
      resultMap.put(key, resultIdx);
      results.add(result);
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

  public Context getActivityContext() {
    return this;
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
          Log.i(TAG, "Warning: You need to grant external storage access to use MLPerf app.");
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
          if (!new File(MLPerfTasks.getLocalPath(model.getSrc())).canRead()) {
            new ModelExtractTask(MLPerfEvaluation.this, mlperfTasks).execute();
            return false;
          }
        }
        DatasetConfig dataset = task.getDataset();
        if (dataset.hasGroundtruthSrc()
            && !new File(MLPerfTasks.getLocalPath(dataset.getGroundtruthSrc())).canRead()) {
          new ModelExtractTask(MLPerfEvaluation.this, mlperfTasks).execute();
          return false;
        }
      }

      setModelIsAvailable();
    }
    return true;
  }

  @Override
  public void onProgressUpdate(int percent) {
    Message message = handler.obtainMessage(MSG_PROGRESS, percent, 0);
    handler.sendMessage(message);
  }

  @Override
  public void onbenchmarkFinished(ResultHolder result) {
    Message message = handler.obtainMessage(MSG_COMPLETE, result);
    handler.sendMessage(message);
  }

  @Override
  public void onAllBenchmarksFinished(float summaryScore, ArrayList<ResultHolder> results) {
    getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
  }

  private void setModelIsAvailable() {
    modelIsAvailable = true;
    Log.i(TAG, "Ready. Click the Run button to evaluate.");
  }

  // ModelExtractTask copies or downloads files to their location (external storage) when
  // they're not available.
  private static class ModelExtractTask extends AsyncTask<Void, String, Void> {
    private final WeakReference<Context> contextRef;
    private final MLPerfConfig mlperfTasks;
    private boolean success = true;
    private String error;

    public ModelExtractTask(Context context, MLPerfConfig mlperfTasks) {
      contextRef = new WeakReference<>(context);
      this.mlperfTasks = mlperfTasks;
    }

    @Override
    protected Void doInBackground(Void... voids) {
      for (TaskConfig task : mlperfTasks.getTaskList()) {
        for (ModelConfig model : task.getModelList()) {
          if (!new File(MLPerfTasks.getLocalPath(model.getSrc())).canRead()) {
            if (!extractFile(model.getSrc())) {
              success = false;
            }
          }
        }
        DatasetConfig dataset = task.getDataset();
        if (isDatasetFileNeedExtract(dataset.getPath())) {
          if (!extractFile(dataset.getPath())) {
            success = false;
          }
        }
        if (isDatasetFileNeedExtract(dataset.getGroundtruthSrc())) {
          if (!extractFile(dataset.getGroundtruthSrc())) {
            success = false;
          }
        }
      }
      return null;
    }

    private boolean isDatasetFileNeedExtract(String path) {
      if (new File(MLPerfTasks.getLocalPath(path)).canRead()) {
        return false;
      }
      return path.startsWith(ASSETS_PREFIX)
          || path.startsWith("http://")
          || path.startsWith("https://");
    }

    private boolean extractFile(String src) {
      String dest = MLPerfTasks.getLocalPath(src);
      File destFile = new File(dest);
      publishProgress(destFile.getName());
      destFile.getParentFile().mkdirs();
      // Extract to a temporary file first, so the app can detects if the extraction failed.
      File tmpFile = new File(dest + ".tmp");
      try {
        InputStream in;
        if (src.startsWith(ASSETS_PREFIX)) {
          AssetManager assetManager = ((MLPerfEvaluation) contextRef.get()).getAssets();
          in = assetManager.open(src.substring(ASSETS_PREFIX.length()));
        } else if (src.startsWith("http://") || src.startsWith("https://")) {
          ConnectivityManager cm =
              (ConnectivityManager) contextRef.get().getSystemService(Context.CONNECTIVITY_SERVICE);
          NetworkInfo activeNetwork = cm.getActiveNetworkInfo();
          boolean isConnected = activeNetwork != null && activeNetwork.isConnectedOrConnecting();
          if (!isConnected) {
            error = "Error: No network connected.";
            return false;
          }
          in = new URL(src).openStream();
        } else {
          in = new FileInputStream(src);
        }
        OutputStream out = new FileOutputStream(tmpFile, /*append=*/ false);
        copyFile(in, out);
        if (MLPerfTasks.isZipFile(src)) {
          if (!unZip(tmpFile, dest)) {
            return false;
          }
          Log.d(TAG, "Unziped " + src + " to " + dest);
        } else {
          tmpFile.renameTo(destFile);
        }
      } catch (IOException e) {
        Log.e(TAG, "Failed to prepare file " + dest + ": " + e.getMessage());
        error = "Error: " + e.getMessage();
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

    private boolean unZip(File inputFile, String dest) {
      InputStream is;
      ZipInputStream zis;
      File destFile = new File(dest);
      destFile.mkdirs();
      try {
        String filename;
        is = new FileInputStream(inputFile);
        zis = new ZipInputStream(new BufferedInputStream(is));
        ZipEntry ze;
        byte[] buffer = new byte[1024];
        int count;

        while ((ze = zis.getNextEntry()) != null) {
          filename = ze.getName();
          // Need to create directories if not exists.
          if (ze.isDirectory()) {
            File fmd = new File(dest, filename);
            fmd.mkdirs();
            continue;
          }

          FileOutputStream fout = new FileOutputStream(new File(dest, filename));
          while ((count = zis.read(buffer)) != -1) {
            fout.write(buffer, 0, count);
          }
          fout.close();
          zis.closeEntry();
        }

        zis.close();
      } catch (IOException e) {
        Log.e(TAG, "Failed to unzip file " + dest + ".zip: " + e.getMessage());
        return false;
      }
      return true;
    }

    @Override
    protected void onPreExecute() {
      Log.i(TAG, "Extracting missing files...");
    }

    @Override
    protected void onProgressUpdate(String... filenames) {
      Log.i(TAG, "Extracting " + filenames[0] + "...");
    }

    @Override
    protected void onPostExecute(Void result) {
      if (success) {
        Log.i(TAG, "All missing files are extracted.");
        ((MLPerfEvaluation) contextRef.get()).setModelIsAvailable();
      } else {
        Log.i(TAG, error);
      }
    }
  }
}
