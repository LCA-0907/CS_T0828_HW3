# CS_T0828_HW3
Code for Selected Topics in Visual Recognition
using Deep Learning(2020 Autumn) HW3

## Hardware
ubuntu 16.04 LTS

Intel® Core™ i9-10900 CPU @ 3.70GHz x 20

RTX 2080 Ti

## Requirements
My Environment settings are below:
* Python = 3.7.9
* pandas = 1.1.3
* opencv = 4.4.0

In this work, I used yolov4 framework from [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet), so you should also followed the instruction of AlexeyAB's darknet:
```
Windows or Linux
CMake >= 3.12
CUDA >= 10.0
OpenCV >= 2.4
cuDNN >= 7.0
```
## Reproducing Submission
To Reproduct the submission, do the folowed steps
1. [Framework Download and Setting](#Framework-Download-and-Setting)
2. [Dataset Preparation](#Dataset-Preparation)
3. [Training](#Training)
4. [Testing](#Testing)

### Framework Download and Setting
This work use [detectron2](https://github.com/facebookresearch/detectron2) to complete instance segmentation task, you can download and install detectron2 by commands:
```
$ git clone https://github.com/facebookresearch/detectron2.git
$ python3 -m pip install -e detectron2
```
Look for more detail use of detectron2, please check its document and tutorial.
### Dataset Preparation
Download tiny VOC dataset from [google drive](https://drive.google.com/drive/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK), unzip and put it in the same directory, it should structured like
```
+- CS_T0828_HW3
|   +- train_images
|   |  2007_000033.jpg
|   |  2007_000042.jpg
|   |  ...
|   +- test_images
|   |  2007_000629.jpg
|   |  2007_001175.jpg
|   |  ...
|
| hw3.py
| util.py
| pascal_train.json
| test.json
```
### Training
To train the model, use command line as
`$ python3 hw3.py --mode train --lr <learning rate> --iter <iteration>`

The default learning rate and iteration are 0.00025 and 30000

After training, check directory output if there is a `model_final.pkl` file which contains the final weight.

### Testing
To test the model, use command
`$ python3 hw3.py --mode test --thres 0.5`
It will generate a `test_out.json` file which contains the testing output in required format.
