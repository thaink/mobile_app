# Prebuilt MLPerf Mobile App #

This directory contains prebuilt versions of the MLPerf (Android) app, for convenience.

## Getting Started ##

Until the app is available in the [Play Store](https://play.google.com/store?hl=en_US), it requires
manual installation from a host machine. note that, due to restrictions in dataset redistribution,
some amount of manual retrieval, processing and packaging is required to use certain task datasets
for accuracy evaluation.

### Requirements ###

 * [Android SDK](https://developer.android.com/studio)
 * Android 4.4+ w/ [USB debugging enabled](https://developer.android.com/studio/debug/dev-options)

### App installation ###

Install the app via `adb` as follows:

```
adb install ${ROOT_DIR}/prebuilt/MLPerf_0.5_beta.apk
```

### Dataset installation ###

While a task dataset isn't required for measuring latency, it is required for measuring accuracy.

This currently requires some manual work to prepare and distribute the relevant datasets. Note that
the full datasets can be extremely large; a subset of the full dataset can be prepared to give a
useful proxy for task-specific accuracy. When prepared, each task directory should be pushed to
`/sdcard/mlperf_datasets/...`.

#### Imagenet (Classifiaction) ####

The app evaluations Image Classification via MobileNet V1
(float & quantized), using any subset of images used in the
[ILSVRC 2012 image classification task](http://www.image-net.org/challenges/LSVRC/2012/).

Expected folder data layout:

*   `imagenet/img` : `folder` \
    Contains the set of images to evaluate over. This could be any subset of the ILSVRC
    2012 evaluation dataset.

*   `imagenet/val.txt` : `txt file` \
    Validation file containing a list of `$IMAGE_NAME $IMAGE_LABEL_INDEX` pairs,
    where `$IMAGE_NAME` is the filanme from the `img/` folder, and `$IMAGE_LABEL_INDEX` is the
    expected label index from inference output.

The classification model files are already in the app, for convenience. Once the data has been
prepared, push the resulting `imagenet` folder to `/sdcard/mlperf_datasets/imagenet`.

#### Coco (Detection) ####

## App Execution ##

After the app and dataset(s) have been installed, simply open `MLPerf Inference` from the Android launcher. The
app UI allows selection of supported tasks, and execution of the supported models (float/quantized)
for each task.


The user can also configure runtime execution with the following options:
 * Number of threads - for CPU execution
 * Delegate (accelerator):
    * None - Default (CPU) path
    * GPU - Enables use of the GPU (via OpenGL/OpennCL)
    * NNAPI - Enables use of [Android's NN API](https://developer.android.com/ndk/guides/neuralnetworks)
