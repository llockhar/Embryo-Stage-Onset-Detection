import numpy as np
import pandas as pd
import argparse
import math


parser = argparse.ArgumentParser()
parser.add_argument("--train_name", type=str, help="Training/Experiment name")
parser.add_argument("--suffix", type=str, help="Folder Suffix")
parser.add_argument("--cross_val", type=bool, help="Whether 5-fold cross-validation was performed", default=False)
args = parser.parse_args()

training_name = args.train_name
suffix = args.suffix

OUT_PATH = training_name + '/Predictions{}/'.format(suffix)
CROSS_VAL = args.cross_val

def read_preds(fold):
    data_df = pd.read_csv(
        OUT_PATH + 'Preds-fold{}.csv'.format(fold), 
        usecols=['Filenames', 'Labels', 'Preds'])
    data_df['Folder'] = [filename.split('/')[0] for filename in data_df['Filenames']]

    return data_df

def _analyze():
    folds = 5 if CROSS_VAL else 1
    for fold in range(folds):
        data_df = read_preds(fold)
        test_folders = data_df.Folder.unique()

        vals = np.zeros((len(test_folders), 6))

        for i,folder in enumerate(test_folders):
            print(folder)
            folder_df = data_df[data_df['Folder'] == folder].reset_index()

            true_morula = folder_df.index[folder_df['Labels'] == 1].min()
            pred_morula = folder_df.index[folder_df['Preds'] == 1].min()
            true_blast = folder_df.index[folder_df['Labels'] == 2].min()
            pred_blast = folder_df.index[folder_df['Preds'] == 2].min()
            if math.isnan(pred_blast):
                print('nan', folder)
                pred_blast = folder_df.index.values.max()

            dif_morula = np.abs(pred_morula - true_morula)
            dif_blast = np.abs(pred_blast - true_blast)

            vals[i,:] = np.array([
                true_morula, pred_morula, true_blast, pred_blast, 
                dif_morula, dif_blast])


        val_df = pd.DataFrame({
            'Folder': test_folders,
            'TrueMorula': vals[:,0],
            'TrueBlast': vals[:,1],
            'PredMorula': vals[:,2],
            'PredBlast': vals[:,3],
            'DifMorula': vals[:,4],
            'DifBlast': vals[:,5]})

        val_df.to_csv(OUT_PATH + 'SeqAnalysis-fold{}.csv'.format(fold), index=False)

if __name__ == '__main__':
  _analyze()
