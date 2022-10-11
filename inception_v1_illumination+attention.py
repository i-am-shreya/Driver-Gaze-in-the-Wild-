#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:55:07 2020

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

import cv2 
import numpy as np 
from keras import backend as K 
from keras.utils import np_utils
from attn_augconv import augmented_conv2d
import math 
from keras.optimizers import SGD 
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD

np.random.seed(1000)

import keras
from keras.layers.core import Layer
import keras.backend as K
import tensorflow as tf

from keras.models import Model
from keras.layers import Conv2D, MaxPool2D,  \
    Dropout, Dense, Input, concatenate,      \
    GlobalAveragePooling2D, AveragePooling2D,\
    Flatten,Activation

import cv2 
import numpy as np 
from keras import backend as k
from keras.utils import np_utils
from attn_augconv import augmented_conv2d
import math 
from keras.optimizers import SGD 
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD

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

kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)

input_layer = Input(shape=(224, 224, 3))

x_i = Conv2D(128, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/i', kernel_initializer=kernel_init_illu, bias_initializer=bias_init)(input_layer)
x_i = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/i')(x_i)

x_o = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
x_o = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x_o)

x = concatenate([x_i, x_o], axis=3)
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
x = augmented_conv2d(x, filters=2000, kernel_size=(3,3),
                         depth_k=0.2, depth_v=0.2,  # dk/v (0.2) * f_out (20) = 4
                         num_heads=4, relative_encodings=True)
#x = MaxPool2D((2, 2), padding='same', strides=(2, 2), name='max_pool_3_2x2/2')(x)

x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)
x = Dense(1024)(x)
x = Activation('relu')(x)
x = Dropout(0.4)(x)
x = Dense(512)(x)
x = Activation('relu')(x)
x_c = Dense(9, activation='softmax', name='output')(x)
#x_r = Dense(9, activation='sigmoid', name='output1')(x)
model = Model(input_layer, x_c, name='inception_v1')

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
adam = keras.optimizers.Adam(lr=0.01, decay=1e-6)

model.compile(loss=['mse'], optimizer=sgd,metrics=['accuracy'])

checkpoint = keras.callbacks.ModelCheckpoint('./models/best_weights.h5', monitor='val_acc', save_best_only=True,
                                              verbose=1,mode='max')

early_stopping=keras.callbacks.EarlyStopping(monitor='val_acc', patience=50, verbose=1, mode='max')

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
#model.save('./models/cardata_inception-v1_0.01_fc_illumination+attention_mse_label-modify.h5')

#%%

import os
import cv2
import keras
import numpy as np
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects



model.load_weights('./models/cardata_inception-v1_0.01_fc_illumination+attention_mse_label-modify.h5')
val_pred=model.predict_generator(validation_generator, steps=len(validation_generator),  verbose=1)
y_true=validation_generator.classes
y_pred=np.argmax(val_pred,axis=1)

pred=model.predict_generator(test_generator, steps=len(test_generator),  verbose=1)
y_true=test_generator.classes
y_pred=np.argmax(pred,axis=1)

#file_val = open('/media/netweb/2.0 TB/shreya/eye_gaze/val_pred_incep_attention+illumination.txt','w') 
#file_val.write('image name'+'  '+'y_true'+'  '+'y_pred'+'\n')

#i = 0
#path='./sets/val_face_modified/'
#for label in range(9):
#    folder_path = path + str(label+1)
#    image_list=os.listdir(folder_path)
#    for j in range(len(image_list)):
#        print(str(i))
#        
#        image_path=folder_path+'/'+image_list[j]
#        image = cv2.imread(image_path)
#        image = cv2.resize(image,(224,224))
#        image = np.reshape(image,(1,224,224,3))
#        pred = model.predict(image)
#        y_true[i]=label  
#        y_pred[i]= np.argmax(pred) 
##        file_val.write(image_list[k]+'       '+str(label+1)+'       '+str(np.argmax(pred)+1)) 
##        file_val.write('\n')
#        i = i+1
        
#file_val.close()   

from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y_true, y_pred)    

from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred)) 
classwise_acc = conf.diagonal()/conf.sum(axis=1)
classwise_acc = classwise_acc*100

#import numpy as np
#import matplotlib.cm as cm
#from vis.visualization import visualize_cam
#import matplotlib.pyplot as plt
#from vis.utils import utils
#
#img = cv2.imread('/media/netweb/2.0 TB/shreya/eye_gaze/data/train/train/1/Sub28_vid_1_frame34_face.png')
#img = np.resize(img,[1,224,224,3])
#
#for modifier in [None, 'guided', 'relu']:
#    # 20 is the imagenet index corresponding to `ouzel`
#    grads = visualize_cam(model, 92, filter_indices=20, 
#                          seed_input=img,  backprop_modifier=modifier)       
#    plt.imshow(grads)
#    plt.show()
