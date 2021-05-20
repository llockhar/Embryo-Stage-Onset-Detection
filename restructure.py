import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os
from os.path import exists

from sklearn.metrics import (
    balanced_accuracy_score, precision_score,
    recall_score, jaccard_score,
    confusion_matrix, classification_report)

parser = argparse.ArgumentParser()
parser.add_argument("--in_path", type=str, help="Path to predicted classifications dataframes")
parser.add_argument("--out_path", type=str, help="Desired output path for restructured dataframes")
parser.add_argument("--loss_func", type=str, help="['MAE', 'NLL'] Type of loss for optimization", default='MAE')
parser.add_argument("--cross_val", type=bool, help="Whether 5-fold cross-validation was performed", default=False)
args = parser.parse_args()

IN_PATH = args.in_path
OUT_PATH = args.out_path
LOSS_FUNC = args.loss_func
CROSS_VAL = args.cross_val

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def read_preds(fold):
    data_df = pd.read_csv(
        IN_PATH + 'Preds-fold{}.csv'.format(fold), 
        usecols=['Filenames', 'Labels', 'Preds', 'Cleavage', 'Morula', 'Blastocyst'])
    data_df['Folder'] = [filename.split('/')[0] for filename in data_df['Filenames']]

    return data_df

def restructure_raw(loss_func):
    folds = 5 if CROSS_VAL else 1
    for fold in range(folds):
        print('-'*50)
        print('...Post-processing sequences...')
        print('-'*50) 

        data_df = read_preds(fold)
        test_folders = data_df.Folder.unique()
        test_filenames = data_df.Filenames.unique()

        vals = np.zeros((len(test_filenames), 2))
        df_index = 0

        for i,folder in enumerate(test_folders):
            print(folder)
            folder_df = data_df[data_df['Folder'] == folder].reset_index()
            seq_len = folder_df.shape[0]
            labels = np.array((folder_df['Labels']))
            preds_in0 = np.expand_dims(np.array(folder_df['Cleavage']), axis=-1)
            preds_in1 = np.expand_dims(np.array(folder_df['Morula']), axis=-1)
            preds_in2 = np.expand_dims(np.array(folder_df['Blastocyst']), axis=-1)
            preds_in = np.concatenate((preds_in0, preds_in1, preds_in2), axis=-1)
            preds_out = np.zeros(folder_df.shape[0])


            min_loss = None
            morula_ind = 1
            blast_ind = 2
            for ind1 in range(1, seq_len-1):
                for ind2 in range(ind1 + 1, seq_len):
                    preds_out_temp = np.zeros((folder_df.shape[0], 3))
                    preds_out_temp[:ind1] = get_one_hot(np.array([0]), 3)
                    preds_out_temp[ind1:] = get_one_hot(np.array([1]), 3)
                    preds_out_temp[ind2:] = get_one_hot(np.array([2]), 3)

                    if loss_func == 'MAE':
                        loss = np.abs((preds_in - preds_out_temp)).sum()
                    elif loss_func == 'NLL':
                        y_pred = np.clip(preds_in, 1e-7, 1 - 1e-7)
                        positive = preds_out_temp * np.log(y_pred)
                        loss = -np.sum(positive, -1).mean()

                    if min_loss is None or loss < min_loss:
                        min_loss = loss
                        morula_ind = ind1
                        blast_ind = ind2

            preds_out[morula_ind:] = 1
            preds_out[blast_ind:] = 2
            
            vals[df_index:df_index+seq_len,0] = labels
            vals[df_index:df_index+seq_len,1] = preds_out

            df_index += seq_len
        
        save_class_results(
            vals[:,0], 
            vals[:,1], 
            np.array(test_filenames),
            fold)


def save_class_results(y_groundtruth, y_predicted, file_names, fold):
    metrics = np.zeros(9)
    metrics_names = [
        'Bal_Acc', 'Prec_Mic', 'Prec_Mac', 'Prec_Wgt', 
        'Rec_Mic', 'Rec_Mac', 'Rec_Wgt', 'Jac_Mac', 'Jac_Wgt'
    ]
    classes = [0, 1, 2]
    class_report_index = pd.Index(
        [classes[0], classes[1], classes[2],\
        'micro avg', 'macro avg', 'weighted avg'
    ])

    y_preds = y_predicted.round()
    y_truth = y_groundtruth.round() 

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
        out_name=out_name
    )

    pred_results = pd.DataFrame({
        "Filenames": file_names,
        "Labels": y_truth,
        "Preds": y_preds,
    })

    class_report = classification_report(y_truth, y_preds, output_dict=True)
    class_report = pd.DataFrame(class_report).transpose()
    class_report = class_report.set_index(class_report_index)
    
    pred_results.set_index("Filenames").to_csv(
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

    if not exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    restructure_raw(LOSS_FUNC)