# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:42:24 2017

@author: workspace
"""

from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
import numpy as np

import keras.backend as K
K.set_image_dim_ordering('th')

imsz = 9

def mymodel():
    model = Sequential() 
    
    model.add(Convolution2D(10, 3, 3, border_mode='same',
                              activation='relu', init='glorot_normal', input_shape=(1,None,None)))

#    model.add(Convolution2D(10, 3, 3, border_mode='same',
#                              activation='relu', init='glorot_normal', input_shape=(1,imsz,imsz)))

    model.add(Convolution2D(10, 3, 3, border_mode='same',
                              activation='relu', init='glorot_normal'))
    
    model.add(Convolution2D(20, 5, 5, border_mode='same',
                              activation='relu', init='glorot_normal'))
    
    model.add(Convolution2D(50, 7, 7, border_mode='same',
                              activation='relu', init='glorot_normal'))
    
    model.add(Convolution2D(20, 7, 7, border_mode='same',
                              activation='relu', init='glorot_normal'))
    
    model.add(Convolution2D(10, 7, 7, border_mode='same',
                              activation='relu', init='glorot_normal'))
    
    model.add(Convolution2D(1, 3, 3, border_mode='same', init='glorot_normal', activation='sigmoid'))
   
    return model

import keras.backend as K
def mymetric(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def myloss(y_true, y_pred):
    return K.mean(K.abs(K.abs(y_pred - y_true)))
   
if __name__ == '__main__':
    # define the model
    model = mymodel()
    model.compile('rmsprop', loss=myloss, metrics=[mymetric]) 
    
    # fitting
    np.random.seed(1337)
    x = np.random.random(size=(10000,1,imsz,imsz))
    y = np.random.random((10000,1,imsz,imsz))
    model.fit(x,y,batch_size=100)
    
    # predict 
    input_array = np.random.random(size=(1,1,600,600)) 
    import time
    tic = time.time()
    output_array = model.predict(input_array)
    toc = time.time()
    
    print(output_array.max())
    print(output_array.min())
    print('total time: %f' %(toc-tic))
    
    import cv2
    cv2.namedWindow('input')
    cv2.imshow('input', input_array[0,0,:,:])
    cv2.namedWindow('output')
    cv2.imshow('output', output_array[0,0,:,:])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    