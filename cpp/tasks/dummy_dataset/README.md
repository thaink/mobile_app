# Dummy dataset

This binary evaluates the performance of tflite for the mlperf benchmark.

## Run the model in performance mode

```bash
blaze run  -c opt --cxxopt='--std=c++11' -- \
cpp/tasks/imagenet_classification/mlperf_dummy \
--model_file=<path to your .tflite model file> \
--num_threads=4 \
--delegate=None \
--output_dir=<choose your output dir>
```

after running this, you will see the performance results logged in the
mlperf_log_summary.txt file in your output dir. You can choose one of the
delegates among gpu and nnapi.
