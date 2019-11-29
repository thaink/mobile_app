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

/** A class that wraps functionality around tflite::mlperf::TfliteMlperfDriver. */
public final class MLPerfDriverWrapper implements AutoCloseable {
  /**
   * MLPerfDriverWrapper constructor is marked as private since the dataset pointer should be hold,
   * managed and deleted by TfliteMlperfDriver. Letting it to be initialized outside this class can
   * lead to various memory management problems.
   */
  private MLPerfDriverWrapper(
      String modelFilePath,
      long datasetHandle,
      int numThreads,
      String delegate,
      int numInput,
      int numOutput) {
    driverHandle =
        nativeInit(modelFilePath, datasetHandle, numThreads, delegate, numInput, numOutput);
  }

  /**
   * {@link runMLPerf} runs a specific model with mlperf.
   *
   * @param mode could be a string of PerformanceOnly, AccuracyOnly or SubmissionRun (both).
   * @param minQueryCount is the minimum number of samples should be run.
   * @param minDurationMs is the minimum duration in ms. After both conditions are met, the test
   *     ends.
   * @param outputDir is the directory to store the log files.
   */
  public void runMLPerf(String mode, int minQueryCount, int minDurationMs, String outputDir) {
    nativeRun(driverHandle, mode, minQueryCount, minDurationMs, outputDir);
  }

  // The latency in ms is formatted with two decimal places.
  public String getLatency() {
    return nativeGetLatency(this.driverHandle);
  }

  // The groundtruth file and format of the accuracy string is up to tasks.
  // Ex: mobilenet image classification on imagenet requires gtFile as imagenet_val.txt and
  // returns accuracy as 12.34%.
  public String getAccuracy(String gtFile) {
    return nativeGetAccuracy(this.driverHandle, gtFile);
  }

  @Override
  public void close() {
    nativeDelete(driverHandle);
  }

  // Native functions.
  private native long nativeInit(
      String modelFilePath,
      long datasetHandle,
      int numThread,
      String delegate,
      int numInput,
      int numOutput);

  private native void nativeRun(
      long driverHandle, String jmode, int minQueryCount, int minDuration, String outputDir);

  private native String nativeGetLatency(long handle);

  private native String nativeGetAccuracy(long handle, String gtFile);

  // Nullness of the pointer is checked inside nativeDelete. Callers can skip that check.
  private native void nativeDelete(long handle);

  // Native functions for dataset manipulation. Nullness of the pointer is checked
  // inside nativeDeleteDataset. Callers can skip that check.
  private static native void nativeDeleteDataset(long handle);

  // Return a pointer of a new Imagenet C++ object.
  private static native long imagenet(String imageDir, int offset, Boolean isRawImages);

  // Return a pointer of a new Coco C++ object.
  private static native long coco(
      String imageDir, int offset, Boolean isRawImages, String groundtruthFile);

  // Return a pointer of a new DummyDataset C++ object.
  private static native long dummyDataset(int numSamples, int inputSize);

  // driverHandle holds a pointer of TfliteMlperfDriver.
  private final long driverHandle;

  /**
   * The Builder class for MLPerfDriverWrapper.
   *
   * <p>The dataset should be set by one of the functions like: useImagenet,...
   */
  public static class Builder implements AutoCloseable {
    private final String modelFilePath;
    private final int numInput;
    private final int numOutput;
    private int numThreads;
    private String delegate;
    private long dataset;

    public Builder(String path, int numInput, int numOutput) {
      modelFilePath = path;
      this.numInput = numInput;
      this.numOutput = numOutput;
      dataset = 0;
    }

    Builder setNumThreads(int numThreads) {
      this.numThreads = numThreads;
      return this;
    }

    Builder setDelegate(String delegate) {
      this.delegate = delegate;
      return this;
    }

    // Offset is used to match ground-truth categories with model output.
    // Some models assume class 0 is background class thus they have offset equals one.
    // The isRawImages is true, the images are expected to be binary files in the .rgb8 extension.
    Builder useImagenet(String imageDir, int offset, Boolean isRawImages) {
      if (dataset != 0) {
        nativeDeleteDataset(dataset);
      }
      dataset = imagenet(imageDir, offset, isRawImages);
      return this;
    }

    Builder useCoco(String imageDir, int offset, Boolean isRawImages, String groundtruthFile) {
      if (dataset != 0) {
        nativeDeleteDataset(dataset);
      }
      dataset = coco(imageDir, offset, isRawImages, groundtruthFile);
      return this;
    }

    Builder useDummy(int numSamples, int inputSize) {
      if (dataset != 0) {
        nativeDeleteDataset(dataset);
      }
      dataset = dummyDataset(numSamples, inputSize);
      return this;
    }

    MLPerfDriverWrapper build() {
      if (dataset == 0) {
        throw new java.lang.IllegalArgumentException("Dataset should be set first");
      }
      // Move the pointer out of dataset. Equivalent to std::move(dataset).
      long datasetHandle = dataset;
      dataset = 0;
      return new MLPerfDriverWrapper(
          modelFilePath, datasetHandle, numThreads, delegate, numInput, numOutput);
    }

    @Override
    public void close() {
      nativeDeleteDataset(dataset);
    }
  }

  static {
    NativeEvaluation.init();
  }
}
