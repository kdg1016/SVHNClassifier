# SVHNClassifier


## Overview

- Multi-Digit Recognition Engine Using Deep Convolution Neural Network Based on Tensorflow.


## Development Environment
- Anaconda 3
- Python 2.7
- Tensorflow - 1.1.0
- h5py

    ```
    In Ubuntu:
    $ sudo apt-get install libhdf5-dev
    $ sudo pip install h5py
    ```
- for DATA_Collect_GUI.py

  - opencv3 - 3.1.0
  - matplotlib
  - imutils


## Setup

1. Clone the source code

    ```
    $ git clone https://github.com/kdg1016/SVHNClassifier    
    $ cd SVHNClassifier
    ```

2. Download [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/) format 1, 
   extract to data folder, now your folder structure should be like below: (extra / test / train)

3. Directory Tree Structure
    ```
    SVHNClassifier
    │
    ├── convert_to_tfrecords.py
    ├── donkey.py
    ├── eval.py
    ├── evaluator.py
    ├── inference.py
    ├── meta.py
    ├── model.py
    ├── train.py
    │
    ├── test_images
    │   │	.jpg
    │   └── ...
    │
    ├── data
    │   ├── Data_Collect_GUI.py
    │   ├── HoonUtils.py
    │   ├── meta.json (Generated through tfrecords converting)
    │   ├── train.tfrecords (Generated through tfrecords converting)
    │   ├── val.tfrecords (Generated through tfrecords converting)
    │   ├── test.tfrecords (Generated through tfrecords converting)
    │   │
    │   ├── extra (Downloaded image directory)
    │   │   ├── 1.png
    │   │   ├── 2.png
    │   │   ├── 3.png
    │   │   ├── ...
    │   │   ├── digitStruct.mat
    │   │   └── see_bboxes.m
    │   ├── test (Downloaded image directory)
    │   │   ├── 1.png
    │   │   ├── 2.png
    │   │   ├── 3.png
    │   │   ├── ...
    │   │   ├── digitStruct.mat
    │   │   └── see_bboxes.m
    │   ├── train (Downloaded image directory)
    │   │   ├── 1.png
    │   │   ├── 2.png
    │   │   ├── 3.png
    │   │   ├── ...
    │   │   ├── digitStruct.mat
    │   │   ├── readme.txt
    │   │   └── see_bboxes.m
    │   └── user_train (User image directory)
    │       ├── 1.png
    │       ├── 2.png
    │       ├── 3.png
    │       ├── ...
    │       ├── see_bboxes.m
    │       └── user_train.csv
    ├── logs (Generated through training)
    │   ├── eval
    │   ├── train
    │   └── ...
    ```


## Usage

1. Convert to TFRecords format


    ```
    $ python convert_to_tfrecords.py
    ```
    
2. Train

    ```
    $ python train.py
    ```
    

3. Evaluate

    ```
    $ python eval.py
    ```
    

4. Visualize(if you need)

    ```
    $ tensorboard --logdir ./logs
    ```
    
5. Try to make an inference

    ```
    $ python inference.py
    (--path_to_image_file : test_images/test1.jpg,  --path_to_restore_checkpoint_file :./logs/train/latest.ckpt)
    ```
 
    ### Samples

    | Training      | Test          |
    |:-------------:|:-------------:|
    | ![Train1](https://github.com/potterhsu/SVHNClassifier/blob/master/images/train1.png?raw=true) | ![Test1](https://github.com/potterhsu/SVHNClassifier/blob/master/images/test1.png?raw=true) |
    | ![Train2](https://github.com/potterhsu/SVHNClassifier/blob/master/images/train2.png?raw=true) | ![Test2](https://github.com/potterhsu/SVHNClassifier/blob/master/images/test2.png?raw=true) |


    ### Inference of outside image
    
    <img src="https://github.com/potterhsu/SVHNClassifier/blob/master/images/inference1.png?raw=true" width="250">


