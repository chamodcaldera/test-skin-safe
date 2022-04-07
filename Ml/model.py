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



#balance Data set samples

def balance_data(train_df, max_samples, min_samples, column, working_dir, image_size):
    train_df = train_df.copy()
    train_df = trim(train_df, max_samples, min_samples, column)
    # make directories to store augmented images
    aug_dir = os.path.join(working_dir, 'aug')
    if os.path.isdir(aug_dir):
        shutil.rmtree(aug_dir)
    os.mkdir(aug_dir)
    for label in train_df['labels'].unique():
        dir_path = os.path.join(aug_dir, label)
        os.mkdir(dir_path)
    # create and store the augmented images
    gen = ImageDataGenerator(horizontal_flip=True, rotation_range=20, width_shift_range=.2,
                             height_shift_range=.2, zoom_range=.2)
    groups = train_df.groupby('labels')  # group by class
    for label in train_df['labels'].unique():  # for looping every class
        group = groups.get_group(label)  # a dataframe holding only rows with the specified label
        sample_count = len(group)  # getting how many samples there are in this class
        if sample_count < max_samples:  # if the class has less than target number of images
            aug_img_count = 0
            delta = max_samples - sample_count  # amount of augmented images to create
            target_dir = os.path.join(aug_dir, label)  # define write folder
            aug_gen = gen.flow_from_dataframe(group, x_col='filepaths', y_col=None, target_size=image_size,
                                              class_mode=None, batch_size=1, shuffle=False,
                                              save_to_dir=target_dir, save_prefix='aug-', color_mode='rgb',
                                              save_format='jpg')
            while aug_img_count < delta:
                images = next(aug_gen)
                aug_img_count += len(images)
    # create aug_df and merge with train_df to create composite training set usc
    aug_fpaths = []
    aug_labels = []
    classlist = os.listdir(aug_dir)
    for class_ in classlist:
        classpath = os.path.join(aug_dir, class_)
        flist = os.listdir(classpath)
        for f in flist:
            fpath = os.path.join(classpath, f)
            aug_fpaths.append(fpath)
            aug_labels.append(class_)
    Fseries = pd.Series(aug_fpaths, name='filepaths')
    Lseries = pd.Series(aug_labels, name='labels')
    aug_df = pd.concat([Fseries, Lseries], axis=1)
    usc = pd.concat([train_df, aug_df], axis=0).reset_index(drop=True)
    print(list(usc['labels'].value_counts()))
    return usc


max_samples = 1006
min_samples = 0
column = 'labels'
working_dir = r'./'
img_size = (300, 300)
usc = balance_data(train_df, max_samples, min_samples, column, working_dir, img_size)

channels = 3
batch_size = 30
img_shape = (img_size[0], img_size[1], channels)
length = len(test_df)
test_batch_size = \
sorted([int(length / n) for n in range(1, length + 1) if length % n == 0 and length / n <= 80], reverse=True)[0]
test_steps = int(length / test_batch_size)
print('test batch size: ', test_batch_size, '  test steps: ', test_steps)


def scal(img):
    return img  #  scaling is not required


trgen = ImageDataGenerator(preprocessing_function=scal, horizontal_flip=True)
tvgen = ImageDataGenerator(preprocessing_function=scal)
train_gen = trgen.flow_from_dataframe(usc, x_col='filepaths', y_col='labels', target_size=img_size,
                                      class_mode='categorical',
                                      color_mode='rgb', shuffle=True, batch_size=batch_size)
test_gen = tvgen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                     class_mode='categorical',
                                     color_mode='rgb', shuffle=False, batch_size=test_batch_size)

valid_gen = tvgen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                      class_mode='categorical',
                                      color_mode='rgb', shuffle=True, batch_size=batch_size)
classes = list(train_gen.class_indices.keys())
class_count = len(classes)
train_steps = int(np.ceil(len(train_gen.labels) / batch_size))

# Base model
modelName = 'EfficientNetB3'
base_model = tf.keras.applications.EfficientNetB2(include_top=False, weights="imagenet", input_shape=img_shape,
                                                  pooling='max')
x = base_model.output
x = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
x = Dense(256, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
          bias_regularizer=regularizers.l1(0.006), activation='relu')(x)
x = Dropout(rate=.45, seed=123)(x)
output = Dense(class_count, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)
model.compile(Adamax(lr=.001), loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 40
patience = 1  # number of epochs to wait to adjust lr if monitored value does not improve
stop_patience = 3  # number of epochs to wait before stopping training if monitored value does not improve
threshold = .9  # if train accuracy is < threshhold adjust monitor accuracy, else monitor validation loss
factor = .5  # factor to reduce lr by
dwell = True  # experimental, if True and monitored metric does not improve on current epoch set  modelweights back to weights of previous epoch
freeze = False  # if true free weights of  the base model
ask_epoch = 5  # inisial number of epoches
batches = train_steps
callbacks = [
    BaseCall(model=model, base_model=base_model, patience=patience, stop_patience=stop_patience, threshold=threshold,
             factor=factor, dwell=dwell, batches=batches, initial_epoch=0, epochs=epochs, ask_epoch=ask_epoch)]

history = model.fit(x=train_gen, epochs=epochs, verbose=0, callbacks=callbacks, validation_data=valid_gen,
                    validation_steps=None, shuffle=False, initial_epoch=0)

tr_plot(history, 0)
sub = 'skin disease'
acc = model.evaluate(test_gen, batch_size=test_batch_size, verbose=1, steps=test_steps, return_dict=False)[1] * 100
msg = f'accuracy on the test set is {acc:5.2f} %'
print(msg)
gen = train_gen
scale = 1
model_save_loc, csv_save_loc = modelSaver(working_dir, model, modelName, sub, acc, img_size, scale, gen)