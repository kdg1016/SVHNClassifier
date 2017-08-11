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
