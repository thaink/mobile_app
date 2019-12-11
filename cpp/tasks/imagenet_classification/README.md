# Image classification on imagenet dataset

This binary evaluates the performance and accuracy of tflite for the mlperf
benchmark. No significant preprocessing is done in this binary, except cropping
and resizing the images.

For performance, Loadgen only measures the delay of model inference excluding
pre-processing. Samples are shuffled before sending for inferencing. You can set
the minimum sample count and minimum duration for performance mode. More sample
count will make the statistics fairer.

## Download the model and dataset

Download the model:

```bash
wget http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz
tar -xvzf mobilenet_v1_1.0_224_quant.tgz
```

Download images (can be downloaded from other sources like
[image-net.org](http://image-net.org/challenges/LSVRC/2012/)) and labels:

```bash
pip install ck --user
ck pull repo:ck-env
ck install package --tags=image-classification,dataset,imagenet,val,original,full #images
ck install package --tags=image-classification,dataset,imagenet,aux #labels
```

## Run the model in performance mode

In this mode, the benchmark will evaluate the performance only.

```bash
blaze run  -c opt --cxxopt='--std=c++11' -- \
cpp/tasks/imagenet_classification/mlperf_image_classification \
--mode=PerformanceOnly \
--model_file=<path to your .tflite model file> \
--images_directory=<path to your images directory> \
--num_threads=8 \
--delegate=None \
--offset=<class offset of the model. Default: 1> \
--output_dir=<choose your output dir>
```

after running this, you will see the performance results logged in the
mlperf_log_summary.txt file in your output dir. You can choose one of the
delegates among gpu and nnapi.

## Run the model in accuracy mode

In contrast to the performance mode, this mode will measure the accuracy only.

```bash
blaze run  -c opt --cxxopt='--std=c++11' -- \
cpp/tasks/imagenet_classification/mlperf_image_classification \
--mode=AccuracyOnly \
--model_file=<path to your .tflite model file> \
--images_directory=<path to your images directory> \
--num_threads=8 \
--delegate=None \
--offset=<class offset of the model. Default: 1> \
--output_dir=<choose your output dir>
```

In order to check the accuracy, you need to use an external script outside
Loadgen:

```bash
wget https://github.com/mlperf/inference/raw/master/v0.5/classification_and_detection/tools/accuracy-imagenet.py
python3 accuracy-imagenet.py --mlperf-accuracy-file=<your output dir>/mlperf_log_accuracy.json \
--imagenet-val-file=<path to the val.txt file> --dtype=int32
```

The accuracy result is printed on the screen.

## Run the tool on Android

(1) Build using the following command:

```bash
blaze build -c opt \
  --config=android_arm64 \
  --cxxopt='--std=c++17' \
  //cpp/tasks/imagenet_classification:mlperf_image_classification
```

(2) Connect your phone. Push the binary to your phone with adb push (make the
directory if required):

```bash
adb push blaze-bin/cpp/tasks/imagenet_classification/mlperf_image_classification /data/local/tmp
adb shell chmod +x /data/local/tmp/mlperf_image_classification
```

(4) Push the TFLite model that you need to test and the images to the device:

```bash
adb push <path to your .tflite model file> /data/local/tmp
adb push <path to your images directory> /data/local/tmp
adb shell mkdir /data/local/tmp/mlperf # Output dir
```

(6) Run the binary.

```bash
adb shell /data/local/tmp/mlperf_image_classification \
--mode=<your desired mode> \
--model_file=<path to your .tflite model file under /data/local/tmp> \
--images_directory=<path to your images directory under /data/local/tmp> \
--num_threads=4 \
--delegate=None \
--offset=<class offset of the model. Default: 1> \
--output_dir=/data/local/tmp/mlperf
```

You can check the output logs in the output dir: /data/local/tmp/mlperf. Pull
them to your pc in order to calculate the accuracy.
