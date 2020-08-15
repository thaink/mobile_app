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

/** Object to hold result parameters for each bechmark */
// TODO : This class would be a good candidate for AutoValue: go/autovalue/builders.
public class ResultHolder {
  private String benchmarkId;
  private String runtime;
  private String benchmarkScore;
  private String benchmarkAccuracy;

  public ResultHolder(String id) {
    benchmarkId = id;
    runtime = "";
    benchmarkScore = "0";
    benchmarkAccuracy = "0";
  }

  public void setRuntime(String runtime) {
    this.runtime = runtime;
  }

  public void setScore(String latency) {
    this.benchmarkScore = latency;
  }

  public void setAccuracy(String accuracy) {
    this.benchmarkAccuracy = accuracy;
  }

  public String getId() {
    return benchmarkId;
  }

  public String getRuntime() {
    return runtime;
  }

  public String getScore() {
    return benchmarkScore;
  }

  public String getAccuracy() {
    return benchmarkAccuracy;
  }

  public void reset() {
    runtime = "";
    benchmarkScore = "0";
    benchmarkAccuracy = "0";
  }
}
