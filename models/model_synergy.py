# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from keras import backend as K
from keras.models import Model
from keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, Concatenate, Lambda, LSTM, Activation)


def vgg16_encoder(input_dim, weights_path=None):
    img_input = Input(shape=(input_dim))

    # Block 1
    x = Conv2D(64,(3, 3),activation=None,padding='same',name='block1_conv1')(img_input)
    x = Conv2D(64,(3, 3),activation='relu',padding='same',name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128,(3, 3),activation='relu',padding='same',name='block2_conv1')(x)
    x = Conv2D(128,(3, 3),activation='relu',padding='same',name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256,(3, 3),activation='relu',padding='same',name='block3_conv1')(x)
    x = Conv2D(256,(3, 3),activation='relu',padding='same',name='block3_conv2')(x)
    x = Conv2D(256,(3, 3),activation='relu',padding='same',name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512,(3, 3),activation='relu',padding='same',name='block4_conv1')(x)
    x = Conv2D(512,(3, 3),activation='relu',padding='same',name='block4_conv2')(x)
    x = Conv2D(512,(3, 3),activation='relu',padding='same',name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512,(3, 3),activation='relu',padding='same',name='block5_conv1')(x)
    x = Conv2D(512,(3, 3),activation='relu',padding='same',name='block5_conv2')(x)
    y = Conv2D(512,(3, 3),activation='relu',padding='same',name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(y)

    model = Model(img_input, x, name='vgg16')
    if weights_path == 'imagenet':
        weights = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        model.load_weights(weights, by_name=True)

    return model

def staging_block(grade, num_outputs):
    def l(input_layer):
        avg = GlobalAveragePooling2D()(input_layer)
        dense = Dense(units=32, activation='relu', name='dense_' + grade)(avg)
        drop = Dropout(rate=0.5, name='drop_' + grade)(dense)
        act_function = 'sigmoid' if num_outputs == 1 else 'softmax'
        out = Dense(num_outputs, activation=act_function, name=grade)(drop)

        return out, avg
    return l


def vgg16_top1(app, img_input, num_outputs):
    x = img_input
    for layer in app.layers[:-3]: 
        layer.trainable = False
    for layer in app.layers[:-1]:
        x = layer(x)
    
    out1, avg = staging_block('EmbryoStage1', num_outputs)(x)

    return out1, avg

def vgg16_top2(app, img_input, num_outputs):
    x = img_input
    for layer in app.layers[:-3]: 
        layer.trainable = False
    for layer in app.layers[:-1]:
        x = layer(x)
    
    out1, avg = staging_block('EmbryoStage2', num_outputs)(x)

    return out1, avg

def buildModel_lstm(input_dim, num_outputs, weights_path=None):
    input_img0 = Input(shape=(input_dim))
    input_img1 = Input(shape=(input_dim))
    # =========================================================================
    vgg16_bottom = vgg16_encoder(input_dim, weights_path)
    x, dense0 = vgg16_top1(vgg16_bottom, input_img0, num_outputs)
    y, dense1 = vgg16_top2(vgg16_bottom, input_img1, num_outputs)
    # =========================================================================
    z = Concatenate()([dense0, dense1])
    z = Dense(units=32, activation='relu', name='dense_Synergy')(z)
    z = Dropout(rate=0.5, name='drop_Synergy')(z)
    z = Dense(1, activation='sigmoid', name='Synergy')(z)
    
    lstm0 = Lambda(lambda i: K.expand_dims(i, axis=0))(x)
    lstm0 = LSTM(num_outputs, return_sequences=True)(lstm0)
    lstm0 = Lambda(lambda i: i[0,:,:])(lstm0)
    lstm0 = Activation('softmax', name="LSTM1")(lstm0)

    lstm1 = Lambda(lambda j: K.expand_dims(j, axis=0))(y)
    lstm1 = LSTM(num_outputs, return_sequences=True)(lstm1)
    lstm1 = Lambda(lambda j: j[0,:,:])(lstm1)
    lstm1 = Activation('softmax', name="LSTM2")(lstm1)
    # =========================================================================
    model = Model(inputs=[input_img0, input_img1], outputs=[x, y, z, lstm0, lstm1])
    
    return model

