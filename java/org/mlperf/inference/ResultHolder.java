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

/** Object to hold result parameters for each model */
// TODO : This class would be a good candidate for AutoValue: go/autovalue/builders.
public class ResultHolder {
  private String model;
  private String runtime;
  private String inferenceLatency;
  private String accuracy;

  public ResultHolder(String model) {
    this.model = model;
    this.inferenceLatency = "0";
    this.accuracy = "0";
  }

  public void setModel(String model) {
    this.model = model;
  }

  public void setRuntime(String runtime) {
    this.runtime = runtime;
  }

  public void setInferenceLatency(String latency) {
    this.inferenceLatency = latency;
  }

  public void setAccuracy(String accuracy) {
    this.accuracy = accuracy;
  }

  public String getModel() {
    return model;
  }

  public String getRuntime() {
    return runtime;
  }

  public String getInferenceLatency() {
    return inferenceLatency;
  }

  public String getAccuracy() {
    return accuracy;
  }

  public void reset() {
    runtime = "";
    inferenceLatency = "0";
    accuracy = "0";
  }
}
