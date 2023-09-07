import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing

from covert import covert

def preprocessing(train, frame, feature):
    if train:
        safe = pd.read_csv('notation/safe_4frame_train.csv', header=None)
        unsafe = pd.read_csv('notation/unsafe_4frame_train.csv', header=None)
    else:
        safe = pd.read_csv('notation/safe_4frame_test.csv', header=None)
        unsafe = pd.read_csv('notation/unsafe_4frame_test.csv', header=None)
    
    safe = covert(safe)
    unsafe = covert(unsafe)
    
    safe = safe[0:(len(safe) // frame) * frame]
    print('safe_shape: ', safe.shape, len(safe) // frame)
    unsafe = unsafe[0:(len(unsafe) // frame) * frame]
    print('unsafe_shape: ', unsafe.shape, len(unsafe) // frame)
    
    # combine dataframe
    all_data = pd.concat([unsafe, safe], axis=0)
    print('all_data_shape: ', all_data.shape)
    del all_data['num']
    
    # normalization
    normalized_data = preprocessing.scale(all_data)
    normalized_data = normalized_data.reshape(int(all_data.shape[0] / frame), frame, feature)
    print('all_data_shape_reshape: ', normalized_data.shape)
    
    # label zero as unsafe data
    zero = pd.DataFrame()
    zeros = np.zeros(len(unsafe) // frame)
    zero['label'] = zeros
    print('unsafe_label_shape: ', zero.shape)
    
    # label one as safe data
    one = pd.DataFrame()
    ones = np.ones(len(safe) // frame)
    one['label'] = ones
    print('safe_label_shape: ', one.shape)
    
    # combine labels
    all_label = pd.concat([zero, one], axis=0)
    print(all_label.shape)
    labels = all_label['label']
    print(labels)
    
    labels = to_categorical(labels)
    x = normalized_data
    y = labels
    
    return x, y
