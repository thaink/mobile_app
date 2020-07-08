# MLPerf datasets

This directory provides implementations of datasets used to evaluate the MLPerf
v0.5 benchmark.These implementations are initially developed to be used with
TFLite. Other backends may need to extend them. The dummy dataset can be used if
you only want to measure the performance.

## Imagenet

The Imagenet dataset can be downloaded from
[image-net.org](http://image-net.org/challenges/LSVRC/2012/). The ground truth
file is
[imagenet_val.txt](../../java/org/mlperf/inference/assets/imagenet_val.txt)
which contains indexes of the corresponding class of each images.

If you want to use a subset of images, remember to use the first N ones.

## COCO

Download the COCO 2017 dataset from
[http://cocodataset.org/#download](http://cocodataset.org/#download) and the
upscale_coco.py from
[https://github.com/mlperf/inference/blob/master/v0.5/tools/upscale_coco](https://github.com/mlperf/inference/blob/master/v0.5/tools/upscale_coco).
Then use the script to process the images:

```bash
python upscale_coco.py --inputs /path-to-coco/ --outputs /output-path/ --size 300 300
```

The ground truth file is
[coco_val.pbtxt](../../java/org/mlperf/inference/assets/coco_val.pbtxt) If you
want to use a subset of images, remember to use the first N images which appear
in the file instances_val2017.json. **Note** that the order of images in this
file and the order of images under the images directory are not the same.

## SQUAD

Download the
[Squad v1.1 evaluation set](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
and the
[vocab file](https://storage.googleapis.com/cloud-tpu-checkpoints/mobilebert/uncased_L-24_H-128_B-512_A-4_F-4_OPT.tar.gz).
Then, you can generate the groundtruth and input tfrecord files for running
inference as below:

1.  Install dependencies:

```
pip install tensorflow tensorflow_hub
```

2.  Download other dependencies from google-research and add its path to
    PYTHONPATH:

```
TMP_DIR=<your tmp directory>
pushd $TMP_DIR
curl -o google-research.tar.gz -L https://github.com/google-research/google-research/archive/256f678d1aeb7a4527031c8dd2f4a2c9f3833f93.tar.gz
tar -xzf google-research.tar.gz --transform s/google-research-256f678d1aeb7a4527031c8dd2f4a2c9f3833f93/google-research/
export PYTHONPATH="${PYTHONPATH}:`pwd`/google-research"
popd
```

3.  Generate tfrecord files for inference:

```
python cpp/datasets/squad_utils/generate_tfrecords.py \
  --vocab_file=<path to vocab.txt> \
  --predict_file=<path to dev-v1.1.json> \
  --output_dir=<output dir> \
  --max_seq_length=384 \
  --doc_stride=128 \
  --max_query_length=64
```

There are default tfrecord files [here](../../java/org/mlperf/inference/assets/)
generated with above default parameters.
