---
layout: post
title: A Dive into Melanoma Classification with Machine Learning
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

The imported Libraries I will be using:

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


