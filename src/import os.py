import os
import IPython.display as ipd
from IPython.display import Image

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


from utils import dataset
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D #, AveragePooling1D
from keras.layers import Flatten, Dropout, Activation # Input, 
from keras.layers import Dense #, Embedding
from keras.utils import np_utils
from tensorflow.keras import optimizers
from sklearn.preprocessing import LabelEncoder

## Data Preparation

dataset_path=os.path.abspath('./Dataset')
destination_path=os.path.abspath("./")
randomize = True
split = 0.8
sample_rate = 20000
emotions=["Anger","disgust","fear","happy","Neutral","Sad","Surprise"]
df, train_df,test_df = dataset.create_and_load_meta_csv_df(dataset_path, destination_path, randomize,split)
print("Direct Samples: ", len(df), "\nTraining Smaples: ", len(train_df), "\nTesting Samples :", len(test_df))
from utils.feature_extraction import get_features_dataframe
from utils.feature_extraction import get_audio_features
trainfeautres = pd.read_pickle("./features_dataframe/trainfeatures")
trainlabel = pd.read_pickle("./features_dataframe/trainlabel")
testfeatures = pd.read_pickle("./features_dataframe/testfeatures")
testlabel = pd.read_pickle("./features_dataframe/testlabel")
trainfeautres.shape
trainfeautres=trainfeautres.fillna(0)
testfeatures=testfeatures.fillna(0)
x_train = np.array(trainfeautres)
y_train = np.array(trainlabel).ravel()
x_test = np.array(testfeatures)
y_test = np.array(testlabel).ravel()
lb = LabelEncoder()
y_train = np.utils.to_categorical(lb.fit_transform(y_train))
