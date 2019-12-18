# MLPerf datasets

This directory provides implementations of datasets used to evaluate the MLPerf
v0.5 benchmark.These implementations are initially developed to be used with
TFLite. Other backends may need to extend them. The dummy dataset can be used if
you only want to measure the performance.

## Imagenet

The Imagenet dataset can be downloaded from
[image-net.org](http://image-net.org/challenges/LSVRC/2012/). The ground truth
file is [imagenet_val.txt](java/org/mlperf/inference/assets/imagenet_val.txt)
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
[coco_val.pbtxt](java/org/mlperf/inference/assets/coco_val.pbtxt) If you want to
use a subset of images, remember to use the first N images which appear in the
file instances_val2017.json. **Note** that the order of images in this file and
the order of images under the images directory are not the same.
