# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import argparse
import os
from os.path import join, exists
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam

from utils.utils import split_folders
from datahandler.read_data import init_data_grade, BlastStageSequence
from models.model_synergy import buildModel_lstm

from sklearn.metrics import (
    balanced_accuracy_score, precision_score,
    recall_score, jaccard_score,
    confusion_matrix, classification_report)


parser = argparse.ArgumentParser()
parser.add_argument("--train_name", type=str, help="Training/Experiment name")
parser.add_argument("--img_path", type=str, help="Path to train/test images")
parser.add_argument("--anno_file", type=str, help="Name of xlsx file containing annotations")
parser.add_argument("--patch_size", type=int, help="Height/Width Image Crop Size", default=320)
parser.add_argument("--batch_size", type=int, help="Training batch size", default=32)
parser.add_argument("--frame_stride", type=int, help="Stride of batches of sequence frames", default=32)
parser.add_argument("--cross_val", type=bool, help="Whether to perform 5-fold cross-validation", default=False)
args = parser.parse_args()


IMG_PATH = args.img_path
ANNO_FILE = args.anno_file
PATCH_SIZE = args.patch_size
BATCH_SIZE = args.batch_size
STRIDE = args.frame_stride
training_name = args.train_name
OUT_PATH = training_name + '/PredictionsTest/'
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

def test_():
    data_df = init_data_grade(ANNO_FILE) 
    folderList = sorted(os.listdir(IMG_PATH))

    folds = 5 if CROSS_VAL else 1

    for fold in range(folds):
        print('-'*30)
        print('...Gathering training and validation data...')
        print('-'*30)   

        _, _, tst_folders, folderList = fetch_folders(folderList, fold)

        if fold >= 0:
            test_gene = BlastStageSequence(
                img_path=IMG_PATH,
                data_df=data_df, 
                folders=tst_folders, 
                batch_size=BATCH_SIZE, 
                stride=STRIDE,
                patch_size=PATCH_SIZE, 
                training=False)
            
            print('-'*30)
            print('...Initializing the classifier network...')
            print('-'*30)    
        
            model = buildModel_lstm(input_dim=(PATCH_SIZE,PATCH_SIZE,3), num_outputs=3)
            model.load_weights(training_name + '/Trainings/fold{}.hdf5'.format(fold))
            model.compile(
                optimizer=Adam(lr=3e-5, amsgrad=True),
                loss=[
                    'categorical_crossentropy',
                    'categorical_crossentropy',
                    'binary_crossentropy',
                    'categorical_crossentropy',
                    'categorical_crossentropy'],
                metrics=['categorical_accuracy'])

            # model.summary()
            print('-'*30)
            print('...Evaluating model...')
            print('-'*30)
            
            [_, _, _, y_pred_labels1, y_pred_labels2] = model.predict_generator(test_gene, verbose=1)
            y_pred_labels = np.concatenate((np.expand_dims(y_pred_labels1, 2), np.expand_dims(y_pred_labels2, 2)), axis=2)
            y_pred_labels = np.squeeze(np.mean(y_pred_labels, axis=2))
            y_true_labels = np.squeeze(test_gene.annos)

            
            save_class_results(
                y_true_labels, 
                y_pred_labels, 
                test_gene.fnames,
                fold)

def save_class_results(y_groundtruth, y_predicted, file_names, fold):
    metrics = np.zeros(9)
    metrics_names = [
        'Bal_Acc', 'Prec_Mic', 'Prec_Mac', 'Prec_Wgt', 
        'Rec_Mic', 'Rec_Mac', 'Rec_Wgt', 'Jac_Mac', 'Jac_Wgt']
    classes = [0, 1, 2]
    class_report_index = pd.Index(
        [classes[0], classes[1], classes[2],\
        'micro avg', 'macro avg', 'weighted avg'])

    y_preds = np.argmax(y_predicted, axis=1)
    y_truth = np.around(y_groundtruth)
    y_raw = []
    for (pred_round, pred_raw) in zip(y_preds, y_predicted.max(axis=1)):
        if pred_round == 0:
            y_raw.append(1 - pred_raw)
        elif pred_round == 1:
            y_raw.append(pred_raw)
        else:
            y_raw.append(1 + pred_raw)
    y_preds_raw = np.array(y_raw)

    pred_results = pd.DataFrame({
        "Filenames": file_names,
        "Labels": y_truth,
        "Preds": y_preds})

    pred_results = pred_results.groupby(['Filenames']).mean().round()
    pred_results.reset_index(level=['Filenames'], inplace=True)

    y_truth = pred_results.iloc[:,1]
    y_truth = y_truth.to_numpy(copy=True)
    y_preds = pred_results.iloc[:,2]
    y_preds = y_preds.to_numpy(copy=True)

    pred_raw_results = pd.DataFrame({
        "Filenames": file_names,
        "Raw Preds": y_preds_raw,
        "Cleavage": y_predicted[:,0],
        "Morula": y_predicted[:,1],
        "Blastocyst": y_predicted[:,2]})
    pred_raw_results = pred_raw_results.groupby(['Filenames']).mean()
    pred_raw_results.reset_index(level=['Filenames'], inplace=True)

    
    metrics[0] = balanced_accuracy_score(y_truth, y_preds)
    metrics[1] = precision_score(y_truth, y_preds, average='micro')
    metrics[2] = precision_score(y_truth, y_preds, average='macro')
    metrics[3] = precision_score(y_truth, y_preds, average='weighted')
    metrics[4] = recall_score(y_truth, y_preds, average='micro')
    metrics[5] = recall_score(y_truth, y_preds, average='macro')
    metrics[6] = recall_score(y_truth, y_preds, average='weighted')
    metrics[7] = jaccard_score(y_truth, y_preds, average='macro')
    metrics[8] = jaccard_score(y_truth, y_preds, average='weighted')
    
    
    cmat = confusion_matrix(y_truth, y_preds)
    out_name = OUT_PATH + "CM-fold{}.png".format(fold)
    plot_confusion_matrix(
        cmat=cmat, 
        classes=classes, 
        out_name=out_name)

    class_report = classification_report(y_truth, y_preds, output_dict=True)
    class_report = pd.DataFrame(class_report).transpose()
    class_report = class_report.set_index(class_report_index)

    pred_results = pred_results.set_index("Filenames").join(pred_raw_results.set_index("Filenames"))
    
    pred_results.to_csv(
        OUT_PATH + "Preds-fold{}.csv".format(fold), 
        index=True)
    class_report.to_csv(
        OUT_PATH + "ClassificationReport-fold{}.csv".format(fold), 
        index=True)

    prerec_results = pd.DataFrame.from_dict(
        {metrics_names[i]:[metrics[i]] for i in range(9)}
    )
    prerec_results.to_csv(
        OUT_PATH + "PrecRec-fold{}.csv".format(fold), 
        index=True)

     
def plot_confusion_matrix(cmat, classes, out_name):
    print(cmat)
    cmap = plt.cm.get_cmap('Blues')
    title = 'Confusion matrix, without normalization'

    fig, ax = plt.subplots()
    im = ax.imshow(cmat, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cmat.shape[1]),
        yticks=np.arange(cmat.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes, yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label'
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cmat.max() / 2.
    for i in range(cmat.shape[0]):
        for j in range(cmat.shape[1]):
            ax.text(j, i, format(cmat[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cmat[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(out_name)
    plt.close()

if __name__ == '__main__':
    test_()
