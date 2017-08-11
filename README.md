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
    ----------------------------------------------------------------------------------------------       
        - data (Downloaded data file)
            - extra
                - 1.png 
                - 2.png
                - ...
                - digitStruct.mat
            - test
                - 1.png 
                - 2.png
                - ...
                - digitStruct.mat
            - train
                - 1.png 
                - 2.png
                - ...
                - digitStruct.mat
    ----------------------------------------------------------------------------------------------       
            - user_train (User add images to directory)
                - 1.png 
                - 2.png
                - ...
                - user_train.csv
        - logs (Generated through training)
            - eval
                - ...
            - train
                - checkpoint
                - latest.ckpt.data...
                - latest.ckpt.index
                - latest.ckpt.meta
                - ...
            - train2
                - ...
            - ...         
    ```
    

