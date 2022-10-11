#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:49:43 2020

@author: fit_staff
"""

import keras
from keras.layers.core import Layer
import keras.backend as K
import tensorflow as tf

from keras.models import Model
from keras.layers import Conv2D, MaxPool2D,  \
    Dropout, Dense, Input, concatenate,      \
    GlobalAveragePooling2D, AveragePooling2D,\
    Flatten,Activation
import os
import cv2 
import numpy as np 
from keras import backend as k 
from keras.utils import np_utils
#from attn_augconv import augmented_conv2d
import math 
from keras.optimizers import SGD 
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
np.random.seed(1000)

def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    
    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    
    return output


def kernel_init_illu(shape, dtype=None):
#    ker = np.zeros(shape, dtype=dtype)
    ker = k.random_normal(shape)*k.constant(value=0.001459, shape=shape, dtype=dtype)
    return ker
        


kernel_init = keras.initializers.glorot_normal()
bias_init = keras.initializers.Constant(value=0.2)

input_layer = Input(shape=(224, 224, 3))

x = Conv2D(128, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init_illu, bias_initializer=bias_init)(input_layer)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=64,
                     filters_3x3_reduce=96,
                     filters_3x3=128,
                     filters_5x5_reduce=16,
                     filters_5x5=32,
                     filters_pool_proj=32,
                     name='inception_3a')

x = inception_module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=128,
                     filters_3x3=192,
                     filters_5x5_reduce=32,
                     filters_5x5=96,
                     filters_pool_proj=64,
                     name='inception_3b')

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=192,
                     filters_3x3_reduce=96,
                     filters_3x3=208,
                     filters_5x5_reduce=16,
                     filters_5x5=48,
                     filters_pool_proj=64,
                     name='inception_4a')



x = inception_module(x,
                     filters_1x1=160,
                     filters_3x3_reduce=112,
                     filters_3x3=224,
                     filters_5x5_reduce=24,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4b')

x = inception_module(x,
                     filters_1x1=128,
                     filters_3x3_reduce=128,
                     filters_3x3=256,
                     filters_5x5_reduce=24,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4c')

x = inception_module(x,
                     filters_1x1=112,
                     filters_3x3_reduce=144,
                     filters_3x3=288,
                     filters_5x5_reduce=32,
                     filters_5x5=64,
                     filters_pool_proj=64,
                     name='inception_4d')


x = inception_module(x,
                     filters_1x1=256,
                     filters_3x3_reduce=160,
                     filters_3x3=320,
                     filters_5x5_reduce=32,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_4e')

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

x = inception_module(x,
                     filters_1x1=256,
                     filters_3x3_reduce=160,
                     filters_3x3=320,
                     filters_5x5_reduce=32,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_5a')

x = inception_module(x,
                     filters_1x1=384,
                     filters_3x3_reduce=192,
                     filters_3x3=384,
                     filters_5x5_reduce=48,
                     filters_5x5=128,
                     filters_pool_proj=128,
                     name='inception_5b')

x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)


x = Dense(1024)(x)
x = Activation('relu')(x)
x = Dropout(0.4)(x)
x = Dense(512)(x)
x = Activation('relu')(x)
#x_c = Dense(9, activation='softmax', name='output1')(x)
x_r = Dense(9, activation='sigmoid', name='output2')(x)
model = Model(input_layer, x_r, name='inception_v1')

model.summary()

train_datagen = keras.preprocessing.image.ImageDataGenerator()#rescale=1./255,shear_range=0.2)
train_generator = train_datagen.flow_from_directory(
    './train_face',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

val_datagen =keras.preprocessing.image.ImageDataGenerator()#rescale=1./255,shear_range=0.2)
validation_generator = val_datagen.flow_from_directory(
    './sets/val_face',
    target_size=(224, 224),
    batch_size=32,shuffle=False,
    class_mode='categorical',
    seed=1)
test_datagen =keras.preprocessing.image.ImageDataGenerator()#rescale=1./255,shear_range=0.2)
test_generator = test_datagen.flow_from_directory(
    './sets/test_face',
    target_size=(224, 224),
    batch_size=32,shuffle=False,
    class_mode='categorical',
    seed=1)

sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss=['mse'], optimizer=sgd,metrics=['accuracy'])

checkpoint = keras.callbacks.ModelCheckpoint('./models/best_weights.h5', monitor='val_accuracy', save_best_only=True,
                                              verbose=1,mode='max')

early_stopping=keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50, verbose=1, mode='max')

#    weight_file = Path('best_weights.h5')
#    if weight_file.is_file():
#        model.load_weights('best_weights.h5')

history=model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=500,verbose=1,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),callbacks=[early_stopping,checkpoint])#, initial_epoch=5)

model.load_weights('./models/best_weights.h5')
val_score=model.evaluate_generator(validation_generator, steps=len(validation_generator),  verbose=1)
test_score=model.evaluate_generator(test_generator, steps=len(test_generator),  verbose=1)
model.save('./models/cardata_inception-v1_0.01_fc_illumination_mse_label_modify.h5')

#%%
import matplotlib.pyplot as plt

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()