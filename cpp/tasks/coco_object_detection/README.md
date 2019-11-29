# Object detection on COCO dataset

This binary evaluates the performance and accuracy of tflite for the mlperf
benchmark on object detection. The images are preprocessed using the
upscale_coco.py script.

For performance, Loadgen only measures the delay of model inference excluding
pre-processing. Samples are shuffled before sending for inferencing. You can set
the minimum sample count and minimum duration for performance mode. More sample
count will make the statistics fairer.

## Download the model and dataset

Download the model:

```bash
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
```

Download the COCO 2017 dataset from
[http://cocodataset.org/#download](http://cocodataset.org/#download) and the
upscale_coco.py from
[https://github.com/mlperf/inference/blob/master/v0.5/tools/upscale_coco](https://github.com/mlperf/inference/blob/master/v0.5/tools/upscale_coco).
Then use the script to process the images:

```bash
python upscale_coco.py --inputs /path-to-coco/ --outputs /output-path/ --size 300 300
```

To prepare the groundtruth pbtxt file used in accuracy calculation, we utilize
the `preprocess_coco_minival` Python binary as follows:

```
blaze run @org_tensorflow//tensorflow/lite/tools/evaluation/tasks/coco_object_detection:preprocess_coco_minival -- \
  --images_folder=/path/to/val2017 \
  --instances_file=/path/to/instances_val2017.json \
  --output_folder=/path/to/output/folder
```

## Run the model in performance mode

In this mode, the benchmark will evaluate the performance only.

```bash
blaze run  -c opt --cxxopt='--std=c++11' -- \
cpp/tasks/coco_object_detection:mlperf_object_detection \
--mode=PerformanceOnly \
--model_file=<path to your .tflite model file> \
--images_directory=<path to your images directory> \
--num_threads=8 \
--delegate=None \
--offset=<class offset of the model. Default: 1> \
--ground_truth_proto=<path to scaled ground_truth.pbtxt> \
--output_dir=<choose your output dir>
```

after running this, you will see the performance results logged in the
mlperf_log_summary.txt file in your output dir. You can choose one of the
delegates among gpu and nnapi.

## Run the model in accuracy mode

In contrast to the performance mode, this mode will measure the accuracy only.

```bash
blaze run  -c opt --cxxopt='--std=c++11' -- \
cpp/tasks/coco_object_detection:mlperf_object_detection \
--mode=AccuracyOnly \
--model_file=<path to your .tflite model file> \
--images_directory=<path to your images directory> \
--num_threads=8 \
--delegate=None \
--offset=<class offset of the model. Default: 1> \
--ground_truth_proto=<path to scaled ground_truth.pbtxt> \
--output_dir=<choose your output dir>
```

In order to check the accuracy, you need to use an external script outside
Loadgen:

```bash
wget https://github.com/mlperf/inference/raw/master/v0.5/classification_and_detection/tools/accuracy-coco.py
python3 accuracy-imagenet.py --mlperf-accuracy-file=<your output dir>/mlperf_log_accuracy.json \
--coco-dir=<path to scaled COCO dir>
```

The accuracy result is printed on the screen.

## Run the tool on Android

(1) Build using the following command:

```bash
blaze build -c opt \
  --config=android_arm64 \
  --cxxopt='--std=c++17' \
  //cpp/tasks/coco_object_detection:mlperf_object_detection
```

(2) Connect your phone. Push the binary to your phone with adb push (make the
directory if required):

```bash
adb push blaze-bin/cpp/tasks/coco_object_detection/mlperf_object_detection /data/local/tmp
adb shell chmod +x /data/local/tmp/mlperf_object_detection
```

(4) Push the TFLite model that you need to test and the images to the device:

```bash
adb push <path to your .tflite model file> /data/local/tmp
adb push <path to your images directory> /data/local/tmp
adb push <path to ground_truth.pbtxt> /data/local/tmp
adb shell mkdir /data/local/tmp/mlperf # Output dir
```

(6) Run the binary.

```bash
adb shell /data/local/tmp/mlperf_object_detection \
--mode=<your desired mode> \
--model_file=<path to your .tflite model file under /data/local/tmp> \
--images_directory=<path to your images directory under /data/local/tmp> \
--num_threads=4 \
--delegate=None \
--offset=<class offset of the model. Default: 1> \
--ground_truth_proto=<path to ground_truth.pbtxt> \
--output_dir=/data/local/tmp/mlperf
```

You can check the output logs in the output dir: /data/local/tmp/mlperf. Pull
them to your pc in order to calculate the accuracy.
