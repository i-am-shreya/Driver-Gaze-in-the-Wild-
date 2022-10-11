#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:32:11 2020

@author: shreya
"""

import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense,Activation, GlobalAveragePooling2D
from keras import backend as K
from keras.utils import to_categorical
import scipy.io as sio
from keras.optimizers import SGD
from keras.layers import Input
import keras
from keras import callbacks
from keras.utils.generic_utils import get_custom_objects

def custom_activation(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'custom_activation': Activation(custom_activation)})

input_tensor = Input(shape=(224,224, 3))
# create the base pre-trained model
base_model = InceptionV3(input_tensor=input_tensor, weights=None, include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(4096)(x)
x= Activation(custom_activation)(x)
#x = Dense(4096)(x)
#x= Activation(custom_activation)(x)
#x = Dense(4096)(x)
#x= Activation(custom_activation)(x)

# and a logistic layer -- let's say we have 200 classes
#predictions = Dense(3, activation='softmax')(x)

predictions = Dense(9, activation='softmax', name='output')(x)
# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
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
model.save('./models/cardata_inception-v3_0.01_fc_illumination+attention_mse_label-modify.h5')
