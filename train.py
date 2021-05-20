# -*- coding: utf-8 -*-
import numpy as np
import argparse
import os
from os.path import exists
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger,
    ReduceLROnPlateau)

from utils.utils import split_folders
from datahandler.read_data import init_data_grade, BlastStageSequence
from models.model_synergy import buildModel_lstm


parser = argparse.ArgumentParser()
parser.add_argument("--train_name", type=str, help="Training/Experiment name")
parser.add_argument("--img_path", type=str, help="Path to train/test images")
parser.add_argument("--anno_file", type=str, help="Name of xlsx file containing annotations")
parser.add_argument("--patch_size", type=int, help="Height/Width Image Crop Size", default=320)
parser.add_argument("--batch_size", type=int, help="Training batch size", default=32)
parser.add_argument("--cross_val", type=bool, help="Whether to perform 5-fold cross-validation", default=False)
args = parser.parse_args()


IMG_PATH = args.img_path
ANNO_FILE = args.anno_file
PATCH_SIZE = args.patch_size
BATCH_SIZE = args.batch_size
training_name = args.train_name
OUT_PATH = training_name + '/Trainings/'
CROSS_VAL = args.cross_val

if not exists(OUT_PATH):
    os.makedirs(OUT_PATH)

K.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def fetch_folders(folderList, fold): 
    train_folders, val_folders, test_folders = split_folders(folderList, fold)
    folderList = np.concatenate((
        train_folders, 
        val_folders, 
        test_folders), axis=0)

    return train_folders, val_folders, test_folders, folderList

def train_():

    data_df = init_data_grade(ANNO_FILE) 
    folderList = sorted(os.listdir(IMG_PATH))

    folds = 5 if CROSS_VAL else 1

    for fold in range(folds):
        print('-'*30)
        print('...Gathering training and validation data...')
        print('-'*30)   

        trn_folders, val_folders, _, folderList = fetch_folders(folderList, fold)

        train_gene = BlastStageSequence(
            img_path=IMG_PATH,
            data_df=data_df, 
            folders=trn_folders, 
            batch_size=BATCH_SIZE, 
            stride=BATCH_SIZE,
            patch_size=PATCH_SIZE, 
            training=True)

        val_gene = BlastStageSequence(
            img_path=IMG_PATH,
            data_df=data_df, 
            folders=val_folders, 
            batch_size=BATCH_SIZE, 
            stride=BATCH_SIZE,
            patch_size=PATCH_SIZE, 
            training=False)
        
        print('-'*30)
        print('...Initializing the classifier network...')
        print('-'*30)    
    
        model = buildModel_lstm(
            input_dim=(PATCH_SIZE,PATCH_SIZE,3), 
            weights_path='imagenet', num_outputs=3)
        model.compile(
            optimizer=Adam(lr=1e-4, amsgrad=True), 
            loss={
                'BlastStage1': 'categorical_crossentropy',
                'BlastStage2': 'categorical_crossentropy',
                'Synergy': 'binary_crossentropy', 
                'LSTM1': 'categorical_crossentropy',
                'LSTM2': 'categorical_crossentropy'},
            metrics={
                'BlastStage1': 'categorical_accuracy',
                'BlastStage2': 'categorical_accuracy',
                'Synergy': 'acc', 
                'LSTM1': 'categorical_accuracy',
                'LSTM2': 'categorical_accuracy'})
        
        
        model_checkpoint = ModelCheckpoint(
            OUT_PATH + 'fold{}.hdf5'.format(fold), 
            monitor='val_loss', 
            save_best_only=True)
        # model.summary()
        print('-'*30)
        print('...Fitting model...')
        print('-'*30)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=15)
        csv_logger = CSVLogger(OUT_PATH + 'logFold{}.csv'.format(fold))
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=8, min_lr=1e-8)

        history = model.fit_generator(
            generator=train_gene,
            epochs = 1000,
            callbacks = [
                model_checkpoint, 
                reduce_lr, 
                early_stop,
                csv_logger],
            validation_data=val_gene,
            verbose=2)
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('Binary Crossentropy')
        plt.xlabel('epoch')
        plt.legend(['Train','Validation'], loc='upper left')
        plt.savefig(OUT_PATH + 'LossCurvefold{}.png'.format(fold))
        plt.clf()


if __name__ == '__main__':
    train_()
