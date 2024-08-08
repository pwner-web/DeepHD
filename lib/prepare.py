import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler

import logging

__author__ = "Gao Shichen"
__mail__ = "gaoshichend@163.com"
__date__ = "2023/5/27"

# Configure logging and set the log level to INFO.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define class of SingleCellDataset
# for convolution
class ConvSingleCellDataset(Dataset):
    # Initialize
    def __init__(self, input_img, labels, size):
        labels = labels.astype('float64')
        self.label = torch.tensor(labels, dtype=torch.int64).squeeze()
        self.img = torch.tensor(input_img).reshape((len(self.label), 1, size))
    
    # Total number of cells
    def __len__(self):
        return len(self.label)
    
    # Index images
    def __getitem__(self, idx):
        img_value = self.img[idx, :]
        label_value = self.label[idx]
        return img_value, label_value

# Define class of SingleCellDataset
# for classic model
class DeepSingleCellDataset(Dataset):
    # Initialize
    def __init__(self, rna, labels):
        self.transcript = torch.tensor(rna, dtype=torch.float)
        labels = labels.astype('float64')
        self.labels = torch.tensor(labels, dtype=torch.float)
    
    # Total number of cells
    def __len__(self):
        return self.transcript.shape[0]
    
    # Index cells
    def __getitem__(self, idx):
        transcript_value = self.transcript[idx, :]
        labels_value = self.labels[idx]
        return transcript_value, labels_value

class PlusScaler(StandardScaler):
    def __init__(self, _array, _Model_):
        self._array = _array
        self._Model_ = _Model_

    def __call__(self,):
        if self._Model_ == "resnet":
            return self._array
        elif self._Model_ == "convnet":
            return np.random.normal(self._array, 1)
        elif self._Model_ == "deepnet":
            return np.random.normal(self._array, 1)

class PrepareprocessingDataset():
    def __init__(self):
        pass
    # Function: determine Gene_id or Gene_name read cut-offs for binary classification 
    def categorize(self, data_all, labels_high, labels_low, ref_col_loc):
        
        high_indices = np.array(data_all.iloc[:,ref_col_loc]==labels_high)
        low_indices = np.array(data_all.iloc[:,ref_col_loc]==labels_low)
        high_count = high_indices.sum()
        low_count = low_indices.sum()
        high_low_indices = np.logical_or(high_indices, low_indices)

        return [high_indices, low_indices, high_low_indices], [high_count, low_count]

    
    # Function: transform data into 1d images
    def img_transform(self, input_data):
        gene_num = input_data.shape[1]
        cell_num = input_data.shape[0]
        final_img = input_data.reshape((cell_num, gene_num, 1))

        return final_img

    def data_structure(self, labels, path, model, labels_high, labels_low, batch_size):
        # Load input file
        input_df = pd.read_csv(path)
        # To create a StandardScaler object
        scaler = StandardScaler()
        # To create a MinMaxScaler object
        scaler = MinMaxScaler()
        # To select the columns for normalization and scaling
        cols_to_scale = [labels]
       
        # To find the column number where the labels is located
        column_index = input_df.columns.get_loc(labels)
        indices, count = self.categorize(input_df, labels_high, labels_low, column_index)

        input_df.loc[indices[0], labels] = 1
        input_df.loc[indices[1], labels] = 0

        input_df = input_df.loc[indices[2], :]
        #y: class array
        y = input_df[labels].values 
        X = input_df.drop(labels, axis=1).iloc[:,1:].values

        X = PlusScaler(X, model).__call__()
        # Split training, validation and test set
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=342, stratify=y)

        # high and low counts in each set
        count_train = [y_train.sum(), len(y_train)-y_train.sum()]
        count_val = [y_val.sum(), len(y_val)-y_val.sum()]
        
        if model == "resnet":
            # Transform data into 1d images
            X_train_img = self.img_transform(X_train)
            X_val_img = self.img_transform(X_val)

            # Create datasets
            img_size = X_train_img[0].shape[0]

            TrainDataSet = ConvSingleCellDataset(X_train_img, y_train, img_size)
            ValDataSet = ConvSingleCellDataset(X_val_img, y_val, img_size)
            return_X_train = X_train_img
        elif model == "convnet":
            # Transform data into 1d images
            X_train_img = self.img_transform(X_train)
            X_val_img = self.img_transform(X_val)

            # Create datasets
            img_size = X_train_img[0].shape[0]

            TrainDataSet = ConvSingleCellDataset(X_train_img, y_train, img_size)
            ValDataSet = ConvSingleCellDataset(X_val_img, y_val, img_size)
            return_X_train = X_train_img
        elif model == "deepnet":
            # Create datasets
            TrainDataSet = DeepSingleCellDataset(X_train, y_train)
            ValDataSet = DeepSingleCellDataset(X_val, y_val)
            return_X_train = X_train

        # Create dataloaders

        train_data_loader = DataLoader(TrainDataSet, batch_size=batch_size, shuffle=True)
        val_data_loader = DataLoader(ValDataSet, batch_size=batch_size, shuffle=True)
        
        return train_data_loader, val_data_loader, count_train, count_val, X, input_df, return_X_train
