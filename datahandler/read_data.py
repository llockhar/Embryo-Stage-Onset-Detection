from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence, to_categorical
from keras_preprocessing.image.utils import img_to_array, load_img
from os.path import join
import random
import pandas as pd 
import numpy as np


def init_data_grade(anno_file, folder_list=None):
    data = pd.read_excel(
        anno_file, usecols=['Folder', 'FullFile', 'EmbryoStage'])

    if folder_list:
        data = data[data['Folder'].isin(folder_list)]
        data = data.reset_index(drop=True)

    return data


class BlastStageSequence(Sequence):
    def __init__(self, 
        img_path, data_df, folders, batch_size, stride, patch_size, training=False
    ):
        self.img_path = img_path
        self.data_df = data_df
        self.folders = folders
        self.batch_size = batch_size
        self.stride = stride
        self.patch_size = patch_size
        self.training = training
        self.datagen = None
        self.batch_counter = 0

        self.filenames = self.data_df.iloc[:,1]
        self.annotations = self.data_df.iloc[:,2]
        self.fnames = None
        self.annos = None
        self.paths = None
        self.get_indices(self.folders)

        if self.training: 
            self.datagen = ImageDataGenerator(
                rotation_range=360,
                width_shift_range=0.10,
                height_shift_range=0.10,
                shear_range=0.05,
                horizontal_flip=True,
                vertical_flip=True)


    def get_indices(self, folderList):
        indices = []
        self.batch_counter = 0
        for folder in folderList:
            folder_df = self.data_df[self.data_df['Folder'] == folder]
            if self.training:
                start_ind = random.randint(0, self.batch_size-1)
                folder_df = folder_df.iloc[start_ind:,:]
            seq_len = folder_df.shape[0]
            folder_inds = list(folder_df.index.values)

            index_counter = 0
            while index_counter + self.batch_size <= seq_len:
                indices += folder_inds[index_counter:index_counter + self.batch_size]
                index_counter += self.stride
                self.batch_counter += 1
            
            if np.mod(seq_len, self.batch_size) != 0 and np.mod(seq_len, self.stride) != 0:
                indices += folder_inds[-self.batch_size:]
                self.batch_counter += 1

            if self.training:
                reset_folder = folder_df.reset_index()

                morula_ind = reset_folder.index[folder_df['BlastStage'] == 1].min()
                blast_ind = reset_folder.index[folder_df['BlastStage'] == 2].min()

                index_counter = morula_ind - np.random.randint(24, 32)
                for _ in range(4):
                    indices += folder_inds[index_counter:index_counter + self.batch_size]
                    index_counter += 8
                    self.batch_counter += 1

                index_counter = blast_ind - np.random.randint(24, 32)
                counter2 = 0
                while index_counter + self.batch_size <= seq_len:
                    if counter2 == 4:
                        break
                    indices += folder_inds[index_counter:index_counter + self.batch_size]
                    index_counter += 8
                    counter2 += 1
                    self.batch_counter += 1
      
        indices = np.array(indices)
        self.fnames = self.filenames[indices]
        self.paths = [join(self.img_path, fname) for fname in self.fnames]
        self.annos = self.annotations[indices]


    def on_epoch_end(self):
        if self.training:
            self.get_indices(self.folders)

    def __len__(self):
        if self.training:
            return (self.batch_counter // 2)
        else:
            return self.batch_counter

    def __getitem__(self, idx): 
        if self.training: 
            batch_fname0 = self.paths[2*idx*self.batch_size:(2*idx+1)*self.batch_size]
            batch_annos0 = self.annos[2*idx*self.batch_size:(2*idx+1)*self.batch_size]
            batch_fname1 = self.paths[(2*idx+1)*self.batch_size:(2*idx+2)*self.batch_size]
            batch_annos1 = self.annos[(2*idx+1)*self.batch_size:(2*idx+2)*self.batch_size]
            
            batch_x0 = np.zeros((self.batch_size,self.patch_size,self.patch_size,3), dtype='float32')
            batch_x1 = np.zeros((self.batch_size,self.patch_size,self.patch_size,3), dtype='float32')

            batch_y = np.zeros(self.batch_size, dtype='float32')
            batch_y0 = np.zeros((self.batch_size, 3), dtype='float32')
            batch_y1 = np.zeros((self.batch_size, 3), dtype='float32')

            for i, (fname0, fname1, anno0, anno1) in enumerate(zip(batch_fname0, batch_fname1, batch_annos0, batch_annos1)):
                img0 = load_img(
                    fname0, color_mode='rgb',
                    target_size=(self.patch_size, self.patch_size),
                    interpolation='nearest')
                img1 = load_img(
                    fname1, color_mode='rgb',
                    target_size=(self.patch_size, self.patch_size),
                    interpolation='nearest')
                x0 = img_to_array(img0, data_format='channels_last')
                x1 = img_to_array(img1, data_format='channels_last')
                # Pillow images should be closed after `load_img`,
                # but not PIL images.
                if hasattr(img0, 'close'):
                    img0.close()
                if hasattr(img1, 'close'):
                    img1.close()
                params = self.datagen.get_random_transform(x0.shape)
                x0 = self.datagen.apply_transform(x0, params)
                x1 = self.datagen.apply_transform(x1, params)
                x0 = self.datagen.standardize(x0)
                x1 = self.datagen.standardize(x1)
                batch_x0[i] = x0
                batch_x1[i] = x1
                batch_y0[i] = to_categorical(anno0, num_classes=3)
                batch_y1[i] = to_categorical(anno1, num_classes=3)
                batch_y[i] = 1. if anno0 == anno1 else 0.

            if np.random.rand() > 0.5:
                for ind in np.random.permutation(np.arange(self.batch_size))[:self.batch_size//4]:
                    l = np.random.beta(0.2, 0.2, 1)
                    X_l = l.reshape(1, 1, 1, 1)
                    Y_l = l.reshape(1, 1)

                    X1 = batch_x0[ind]
                    X2 = batch_x1[ind]
                    batch_x0[ind] = X1 * X_l + X2 * (1 - X_l)

                    Y1 = batch_y0[ind]
                    Y2 = batch_y1[ind]
                    batch_y0[ind] = Y1 * Y_l + Y2 * (1 - Y_l)
            
            return [batch_x0, batch_x1], [batch_y0, batch_y1, batch_y, batch_y0, batch_y1]
        else:
            batch_fname = self.paths[idx*self.batch_size:(idx+1)*self.batch_size]
            batch_annos = self.annos[idx*self.batch_size:(idx+1)*self.batch_size]
        
            batch_x = np.zeros((self.batch_size,self.patch_size,self.patch_size,3), dtype='float32')
            batch_y = np.ones(self.batch_size, dtype='float32')
            batch_anno = np.zeros((self.batch_size, 3), dtype='float32')

            for i, (fname, anno) in enumerate(zip(batch_fname, batch_annos)):
                img = load_img(fname,
                            color_mode='rgb',
                            target_size=(self.patch_size, self.patch_size),
                            interpolation='nearest')
                x = img_to_array(img, data_format='channels_last')
                # Pillow images should be closed after `load_img`,
                # but not PIL images.
                if hasattr(img, 'close'):
                    img.close()
                if self.datagen:
                    params = self.datagen.get_random_transform(x.shape)
                    x = self.datagen.apply_transform(x, params)
                    x = self.datagen.standardize(x)
                batch_x[i] = x
                batch_anno[i] = to_categorical(anno, num_classes=3)
            
            return [batch_x, batch_x], [batch_anno, batch_anno, batch_y, batch_anno, batch_anno]


