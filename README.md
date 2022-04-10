# YOLO-tf2

A Tensorflow-2.0 version implementation of YOLO. <br />
Joseph Redmon, Santosh Kumar Divvala, Ross B. Girshick, Ali Farhadi: You Only Look Once: Unified, Real-Time Object Detection in CVPR 2016.

[arxiv](https://arxiv.org/pdf/1506.02640.pdf)

## Architecture

![](https://github.com/jaroslaw1007/YOLO-tf2/blob/main/Architecture-of-YOLO-CNN.png)

## Dependencies
* [Tensorflow2](https://www.tensorflow.org) >= 2.0.0

## Datasets

We use [PASCOL VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) as our training and testing datasets.
You can download data with this [URL](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar).

## Augmentation

We use the augmentation methods refer to the [repo](https://github.com/Paperspace/DataAugmentationForObjectDetection.git).
Including random scale, random rotation, random color distortion, random noise, random translation, random flip.

## Installation

```
git clone https://github.com/Paperspace/DataAugmentationForObjectDetection.git
```

## Training

First you need to go config.py to fill in the directories you place the training and testing images.

```
IMAGE_DIR = ...
IMAGE_DIR_TEST = ...
```

Then

```
python main.py
``` 

## Demo

Blur the image for privacy.

![](https://github.com/jaroslaw1007/YOLO-tf2/blob/main/demo.png)

## Citing

```
@article{DBLP:journals/corr/RedmonDGF15,
  author    = {Joseph Redmon and
               Santosh Kumar Divvala and
               Ross B. Girshick and
               Ali Farhadi},
  title     = {You Only Look Once: Unified, Real-Time Object Detection},
  journal   = {CoRR},
  volume    = {abs/1506.02640},
  year      = {2015},
  url       = {http://arxiv.org/abs/1506.02640},
  eprinttype = {arXiv},
  eprint    = {1506.02640},
  timestamp = {Mon, 13 Aug 2018 16:48:08 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/RedmonDGF15.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
