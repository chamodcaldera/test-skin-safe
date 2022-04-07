import cv2
import tensorflow as tf
from PIL.ImageOps import crop
#from keras.metrics import acc
from keras.metrics import acc
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, Sequential

import numpy as np
import pandas as pd
import shutil
import time
# import cv2 as cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import os
import seaborn as sns
# sns.set_style('darkgrid')
from PIL import Image
# from sklearn.metrics import confusion_matrix, classification_report
# from IPython.core.display import display, HTML
# stop annoying tensorflow warning messages
import logging



logging.getLogger("tensorflow").setLevel(logging.ERROR)



#preprocess the data set

def preprocessdata(sdir, trsplit, vsplit):
    file_paths = []
    labels = []
    classlist = os.listdir(sdir)
    for klass in classlist:
        classpath = os.path.join(sdir, klass)
        flist = os.listdir(classpath)
        for f in flist:
            fpath = os.path.join(classpath, f)
            file_paths.append(fpath)
            labels.append(klass)
    Fseries = pd.Series(file_paths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis=1)
    # split dataframe into train_df and test_df
    dsplit = vsplit / (1 - trsplit)
    strat = df['labels']
    train_df, dummy_df = train_test_split(df, train_size=trsplit, shuffle=True, random_state=123, stratify=strat)
    strat = dummy_df['labels']
    valid_df, test_df = train_test_split(dummy_df, train_size=dsplit, shuffle=True, random_state=123, stratify=strat)
    print('train_df length: ', len(train_df), '  test_df length: ', len(test_df), '  valid_df length: ', len(valid_df))
    print(train_df['labels'].value_counts())
    return train_df, test_df, valid_df


#dataset path input
sdir = r'C:\Users\c_chamodkaldera\Downloads\archive\IMG_CLASSES'
train_df, test_df, valid_df = preprocessdata(sdir, .8, .1)
