# yolov3-ios
Using yolo v3 object detection on ios platform.

## Example applications:
![car](https://raw.githubusercontent.com/Mrlawrance/yolov3-ios/master/imgfolder/car.jpeg)

## QuickStart:
Build tiny_model.xcodeproj in "ios".

## Do it yourself
### Training
The training process mainly consults [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3). We add yolov3 with [Densnet](https://arxiv.org/pdf/1608.06993.pdf).

#### 0.Requirement
* python 3.6.4
* keras 2.1.5
* tensorflow 1.6.0

#### 1.Generate datasets
Generate datasets with VOC format. And try <p><a href="/home" title=" ">python voc_annotation.py</a></p>
