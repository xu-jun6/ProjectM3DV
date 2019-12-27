import random
import tensorflow as tf
import pandas as pd      

import keras
from keras.models import Model
from keras.layers import Input, Convolution3D, MaxPooling3D, Flatten, Dropout,AveragePooling3D, BatchNormalization, Activation, Dense
from keras.metrics import binary_accuracy, binary_crossentropy
from keras.optimizers import Adam, RMSprop,Adadelta
from keras.losses import categorical_crossentrop


def 3D_NET(input_shape = (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1), 
                load_weight_path = None, USE_DROPOUT = None) -> Model:
    inputs = Input(shape = input_shape, name = "input_1")

    
    x = inputs


    x = Convolution3D(16, 3, 3, 3, border_mode = 'same', name = 'conv1a', subsample = (1, 1, 1))(x)
    x = Activation('relu')(x)
    x = Convolution3D(16, 3, 3, 3, border_mode = 'same', name = 'conv1b', subsample = (1, 1, 1))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool1')(x)

    
    x = Convolution3D(32, 3, 3, 3, border_mode = 'same', name = 'conv2a', subsample = (1, 1, 1))(x)
    x = Activation('relu')(x)
    x = Convolution3D(32, 3, 3, 3, border_mode = 'same', name = 'conv2b', subsample = (1, 1, 1))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2), border_mode = 'valid', name = 'pool2')(x)


    x = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same', name='conv3a', subsample=(1, 1, 1))(x)
    x = Activation('relu')(x)
    x = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same', name='conv3b', subsample=(1, 1, 1))(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2), border_mode = 'valid', name = 'pool3')(x)
    x = Dropout(p = 0.2)(x)

    
    x = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same', name='conv4a', subsample=(1, 1, 1))(x)
    x = Activation('relu')(x)
    x = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same', name='conv4b', subsample=(1, 1, 1))(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2), border_mode = 'valid', name = 'pool4')(x)


    x = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv5a', subsample=(1, 1, 1))(x)
    x = Activation('relu')(x)
    x = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv5b', subsample=(1, 1, 1))(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2), border_mode = 'valid', name = 'pool5')(x)
	
        
    x = Convolution3D(64, 2, 2, 2, activation = "relu", name = "last_64")(x)
    x = Activation('relu')(x)
    x = Flatten(name = "outclass")(x)
    x = Dense(512, activation = 'relu')(x)
    x = Dropout(p = 0.2)(x)
    x = Dense(2, activation = 'softmax')(x)
    model = Model(input = inputs, output = [x])
    model.compile(optimizer = Adam(lr=0.0001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0),
                  loss = binary_crossentropy,
                  metrics = ['binary_accuracy'])
    
    return model





















