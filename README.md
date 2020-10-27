# tian-cthit-Mask-RCNN-and-Finetuning-VGG16

This repository includes implementation of Mash-RCNN and Finetuning of clissfication networks such as [VGG11](https://arxiv.org/pdf/1409.1556.pdf). The code is developed based on [Pytorch official tutorials](https://pytorch.org/tutorials/)

## Mask-RCNN 
[Mask-RCNN](https://arxiv.org/abs/1703.06870), which is based on top of [Faster R-CNN](https://arxiv.org/abs/1506.01497). Faster R-CNN is a model that predicts both bounding boxes and class scores for potential objects in the image.

Mask-RCNN architecture:
![Mask-RCNN](https://github.com/tian-cthit/tian-cthit-Mask-RCNN-and-Finetuning-VGG16/blob/main/Mask-RCNN/figures/tv_image03.png)

Mask R-CNN adds an extra branch into Faster R-CNN, which also predicts segmentation masks for each instance.

![Mask-RCNN2](https://github.com/tian-cthit/tian-cthit-Mask-RCNN-and-Finetuning-VGG16/blob/main/Mask-RCNN/figures/tv_image04.png)

Here is one example of a pair of images and segmentation masks:

![example](https://github.com/tian-cthit/tian-cthit-Mask-RCNN-and-Finetuning-VGG16/blob/main/Mask-RCNN/figures/tv_image01.png)

