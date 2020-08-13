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

import android.util.Log;
import java.util.HashMap;
import org.mlperf.proto.ModelConfig;

/* BenchMark represents the benchmark of a model. */
final class Benchmark {
  public static final String TAG = "Benchmark";
  // Value that corresponds to MLPerf “speedometer” being at max.
  public static final float SUMMARY_SCORE_MAX = 4000;
  // Map from benchmark id to its score max.
  private static final HashMap<String, Integer> iconMap =
      new HashMap<String, Integer>() {
        {
          put("IC_tpu_uint8", R.id.action_play);
          put("IC_tpu_float32", R.id.action_play);
          put("IC_tpu_uint8_offline", R.id.action_play);
          put("IC_tpu_float32_offline", R.id.action_play);
          put("OD_uint8", R.id.action_play);
          put("OD_float32", R.id.action_play);
          put("LU_int8", R.id.action_play);
          put("LU_float32", R.id.action_play);
          put("LU_gpu_float32", R.id.action_play);
          put("LU_nnapi_int8", R.id.action_play);
          put("IS_int8", R.id.action_play);
          put("IS_uint8", R.id.action_play);
          put("IS_float32", R.id.action_play);
        }
      };

  // The config associated with this Benchmark.
  private ModelConfig modelConfig;

  public Benchmark(ModelConfig model) {
    if (!iconMap.containsKey(model.getId())) {
      Log.e(TAG, "benchmarkId not in iconMap");
    }
    modelConfig = model;
  }

  public String getId() {
    return modelConfig.getId();
  }

  public String getName() {
    return modelConfig.getName();
  }

  public int getIcon() {
    return iconMap.get(modelConfig.getId());
  }

  public float getScoreMax() {
    return modelConfig.getScoreMax();
  }
}
