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