import os
from library import use_gpu, num_of_terms, scaling_factor, input_x, input_y, image_disect, train_total, validation_total, label_dir, image_dir
if use_gpu == False:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import re
import sys
from tensorflow import keras

train_files = []
for i in range(train_total + validation_total):
    train_files.append('image_{}.npy'.format(i))

partition = {
              'train': train_files[:train_total],
              'validation': train_files[train_total:]
            }

np.random.shuffle(partition['train'])

# Parameters
params = {'dim': (input_x, input_y),
          'batch_size': 32,
          'shuffle': True}


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(input_x, input_y),
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        #print(list_IDs_temp)

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        #print(y[0][:10])
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_label(self, string):
        is_num = 0
        char_list = ''
        for char in string:
            if is_num == 1:
                if char != '.':
                    char_list = char_list + char
                else:
                    break
            if char == '_':
                is_num = 1
        return int(char_list)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 2))
        y = np.empty((self.batch_size, num_of_terms - 2), dtype='float64')
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            tmp = image_disect(np.load(image_dir + '\\' + ID, allow_pickle=True))
            # Adding random noise to image
            tmp = tmp + np.random.uniform(-0.025, 0.025, tmp.shape)
            tmp = np.clip(tmp, 0, 1)
            # Random translation of image
            tmp = np.roll(tmp, np.random.randint(-5, 5, 2), (0, 1))
            X[i,] = tmp
            label_data_path = os.path.join(label_dir + '\coefficient_{}.npy'.format(self.get_label(ID)))
            label_data = np.load(label_data_path, allow_pickle=True)
            y[i] = label_data.dot(scaling_factor)[2 :num_of_terms]
        return X, y

training_generator = DataGenerator(partition['train'], **params)
validation_generator = DataGenerator(partition['validation'], **params)