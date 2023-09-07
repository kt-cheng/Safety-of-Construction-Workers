import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing

from covert import covert

def pre(train, frame, feature):
    if train:
        safe = pd.read_csv('notation/safe_4frame_train.csv', header=None)
        unsafe = pd.read_csv('notation/unsafe_4frame_train.csv', header=None)
    else:
        safe = pd.read_csv('notation/safe_4frame_test.csv', header=None)
        unsafe = pd.read_csv('notation/unsafe_4frame_test.csv', header=None)
    
    # new add
    safe = covert(safe)
    unsafe = covert(unsafe)
    
    # end
    safe = safe[0:(len(safe) // frame) * frame]
    print('safe_shape:', safe.shape, len(safe) // frame)
    unsafe = unsafe[0:(len(unsafe) // frame) * frame]
    print('unsafe_shape:', unsafe.shape, len(unsafe) // frame)
    
    # combine dataframe
    alldata = pd.concat([unsafe, safe], axis=0)
    print('all_data_shape:', alldata.shape)
    del alldata['num']
    
    # normalization
    dataa = preprocessing.scale(alldata)
    dataa = dataa.reshape(int(alldata.shape[0] / frame), frame, feature)
    print('all_data_shape_reshape:', dataa.shape)
    
    # label combine
    # add labels to safe data
    zero = pd.DataFrame()
    a_0 = np.zeros(len(unsafe) // frame)
    zero['label'] = a_0
    print('unsafe_label_shape', zero.shape)
    
    one = pd.DataFrame()
    a_1 = np.ones(len(safe) // frame)
    one['label'] = a_1
    print('safe_label_shape', one.shape)
    
    alldata_label = pd.concat([zero, one], axis=0)
    print(alldata_label.shape)
    labels = alldata_label['label']
    print(labels)
    
    labels = to_categorical(labels)
    x = dataa
    y = labels
    
    return x, y
