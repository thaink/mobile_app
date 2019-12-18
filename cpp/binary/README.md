# MLPerf backends

This directory provides a main binary file which can be used to evaluate the
mlperf benchmark with a specific set of (backend, dataset).

To set the dataset and backend used by the binary, please add
`--cxxopt='-DBACKEND=<your backend>' --cxxopt='-DDATASET=<your dataset>'` in the
bazel build or run command. For example, you can use the following command to
run TFLite with the dummy dataset:

```bash
bazel run  -c opt --cxxopt='--std=c++14' \
  --cxxopt='-DBACKEND=TFLITE' --cxxopt='-DDATASET=DUMMY' -- \
  cpp/binary:main \
  --mode=SubmissionRun \
  --model_file=<path to the model file> \
  --num_threads=4 \
  --num_inputs=<number of inputs> \
  --num_outputs=<number of outputs> \
  --delegate=None \
  --output_dir=<mlperf output directory>
```

Each set of (backend, dataset) has a different set of arguments, so please use
`--help` argument to check which flags are available. Ex:

```bash
bazel run  -c opt --cxxopt='--std=c++14' \
  --cxxopt='-DBACKEND=TFLITE' --cxxopt='-DDATASET=DUMMY' -- \
  cpp/binary:main --help
```

The supported backends and datasets for this binary is listed in the enum
BackendType and DatasetType in main.cc.
