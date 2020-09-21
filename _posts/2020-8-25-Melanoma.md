---
layout: post
title: A Dive into Detecting Melanomas with Machine Learning
image: /assets/images/Melanoma/Living-or-Coping-with-Cancer-Ribbon.png
---

Skin cancer is the most prevalent type of cancer. Melanoma, specifically, is responsible for 75% of skin cancer deaths, despite being the least common skin cancer. The American Cancer Society estimates over 100,000 new melanoma cases will be diagnosed in 2020. It's also expected that almost 7,000 people will die from the disease. As with other cancers, early and accurate detection—potentially aided by data science—can make treatment more effective.

In this competition I competed in, we need to identify melanoma in images of skin lesions. This is a very challenging image classification task as seen by looking at the sample images below. Can you recognize the differences between images? Below are example of skin images with and without melanoma.

Examples WITH Melanoma

![Example WITH Melanoma](/assets/images/Melanoma/WXitrw2.png){:class="img-responsive"}

Examples WITHOUT Melanoma

![Example WITHOUT Melanoma](/assets/images/Melanoma/tmZmY8H.png){:class="img-responsive"}

For this specific competition, we are going to use Stratified KFold cross-validator method for predicting our data.

## What is KFold?
Cross-validaation is a resampling procedure used to evaluate machine learning models on a limited data sample. This procedure has a single parameter called K that refers to the number opf groups that a given data sample is to be split into. As such, the procedure is often called k-fold cross-validation. So what is stratification? Stratification refers to the process of rearranging data as to ensure each fold is a good representation of the whole. So when we put these together, a Stratified KFold is when your model shuffles your data then splits that daata into n_splits.

The imported libraries I will be using for this competition:

```python
import pandas as pd, numpy as np
from kaggle_datasets import KaggleDatasets
import tensorflow as tf, re, math
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
```

In order to be a proper cross validation with a meaningful overall CV score (aligned with LB score), you need to choose the same IMG_SIZES, INC2019, INC2018, and EFF_NETS for each fold. If your goal is to just run lots of experiments, then you can choose to have a different experiment in each fold. Then each fold is like a holdout validation experiment.

    - DEVICE - is GPU or TPU
    - SEED - a different seed produces a different triple stratified kfold split.
    - FOLDS - number of folds. Best set to 3, 5, or 15 but can be any number between 2 and 15
    - IMG_SIZES - is a Python list of length FOLDS. These are the image sizes to use each fold
    - INC2019 - This includes the new half of the 2019 competition data. The second half of the 2019 data is the comp data from 2018 plus 2017
    - INC2018 - This includes the second half of the 2019 competition data which is the comp data from 2018 plus 2017
    - BATCH_SIZES - is a list of length FOLDS. These are batch sizes for each fold. For maximum speed, it is best to use the largest batch size your GPU or TPU allows.
    - EPOCHS - is a list of length FOLDS. These are maximum epochs. Note that each fold, the best epoch model is saved and used. So if epochs is too large, it won't matter.
    - EFF_NETS - is a list of length FOLDS. These are the EfficientNets to use each fold. The number refers to the B. So a number of 0 refers to EfficientNetB0, and 1 refers to  EfficientNetB1, etc.
    - WGTS - this should be 1/FOLDS for each fold. This is the weight when ensembling the folds to predict the test set. If you want a weird ensemble, you can use different weights.
    - TTA - test time augmentation. Each test image is randomly augmented and predicted TTA times and the average prediction is used. TTA is also applied to OOF during validation.

This ended up being my configuration:

```python
DEVICE = "TPU" #or "GPU"

# USE DIFFERENT SEED FOR DIFFERENT STRATIFIED KFOLD
SEED = 42

# NUMBER OF FOLDS. USE 3, 5, OR 15 
FOLDS = 5

# WHICH IMAGE SIZES TO LOAD EACH FOLD
# CHOOSE 128, 192, 256, 384, 512, 768 
IMG_SIZES = [384,384,384,384,384]

# INCLUDE OLD COMP DATA? YES=1 NO=0
INC2019 = [0,0,0,0,0]
INC2018 = [1,1,1,1,1]

# BATCH SIZE AND EPOCHS
BATCH_SIZES = [32]*FOLDS
EPOCHS = [12]*FOLDS

# WHICH EFFICIENTNET B? TO USE
EFF_NETS = [6,6,6,6,6]

# WEIGHTS FOR FOLD MODELS WHEN PREDICTING TEST
WGTS = [1/FOLDS]*FOLDS

# TEST TIME AUGMENTATION STEPS
TTA = 11
```

Since I will be using a TPU (Tensor Processing Unit) to run my model, I'll save my configurations on something called TFRecords. 

```python
if DEVICE == "TPU":
    print("connecting to TPU...")
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        print("Could not connect to TPU")
        tpu = None

    if tpu:
        try:
            print("initializing  TPU ...")
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
            print("TPU initialized")
        except _:
            print("failed to initialize TPU")
    else:
        DEVICE = "GPU"

if DEVICE != "TPU":
    print("Using default strategy for CPU and single GPU")
    strategy = tf.distribute.get_strategy()

if DEVICE == "GPU":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    

AUTO     = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')
```