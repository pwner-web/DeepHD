import pandas as pd
import numpy as np
import torch
#from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#import hiddenlayer as h

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

class PrepareprocessingDataset():
    def __init__(self):
        pass
    # Function: determine Gene_id or Gene_name read cut-offs for binary classification 
    def categorize(self, data_all, labels_high, labels_low, ref_col_loc):
        #data_Gene = data_all.iloc[:,ref_col_loc]
        #print(data_Gene)
        #cutoff = np.quantile(data_Gene, [quantile_high, quantile_low], interpolation="nearest").tolist()
        #cutoff = np.quantile(data_all.iloc[:,ref_col_loc], [quantile_high, quantile_low], interpolation="lower").tolist()
        #cutoff = np.quantile(data_all.iloc[:,ref_col_loc], [quantile_high, quantile_low], interpolation="linear").tolist()
        #print(cutoff)
        #logging.info("Cut-off for {Gene} high: {cutoff_0}; Cut-off for {Gene} low: {cutoff_1}".format(Gene=Gene, cutoff_0=cutoff[0], cutoff_1=cutoff[1]))

        high_indices = np.array(data_all.iloc[:,ref_col_loc]==labels_high)
        low_indices = np.array(data_all.iloc[:,ref_col_loc]==labels_low)
        high_count = high_indices.sum()
        low_count = low_indices.sum()
        # Make the data of two labels equal.
        #if high_count > low_count:
        #    random_indices = np.random.choice(np.where(high_indices)[0], size=high_count - low_count, replace=False)
        #    high_indices[random_indices] = False
        #elif low_count > high_count:
        #    random_indices = np.random.choice(np.where(low_indices)[0], size=low_count - high_count, replace=False)
        #    low_indices[random_indices] = False
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
        # To normalize and scale the selected columns
        #input_df[cols_to_scale] = scaler.fit_transform(input_df[cols_to_scale])

        # To find the column number where the labels is located
        column_index = input_df.columns.get_loc(labels)
        # Process data: binary classification
        # divided into upper and lower quartiles
        #quantile_high, quantile_low = cut_off_high, cut_off_low
        indices, count = self.categorize(input_df, labels_high, labels_low, column_index)

        input_df.loc[indices[0], labels] = 1
        input_df.loc[indices[1], labels] = 0
        #print(input_df.head())

        input_df = input_df.loc[indices[2], :]
        #print(input_df.head())
        #y: class array
        y = input_df[labels].values 
        #X: transcript data array
        #X = input_df.iloc[:, 1:-1].values
        X = input_df.drop(labels, axis=1).iloc[:,1:].values
        # Split training, validation and test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.1, random_state=342, stratify=y)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=2, stratify=y_train_val)

        # high and low counts in each set
        count_train = [y_train.sum(), len(y_train)-y_train.sum()]
        count_val = [y_val.sum(), len(y_val)-y_val.sum()]
        count_test = [y_test.sum(), len(y_test)-y_test.sum()]

        if model == "resnet":
            # Transform data into 1d images
            X_train_img = self.img_transform(X_train)
            X_val_img = self.img_transform(X_val)
            X_test_img = self.img_transform(X_test)

            # Create datasets
            img_size = X_train_img[0].shape[0]

            TrainDataSet = ConvSingleCellDataset(X_train_img, y_train, img_size)
            ValDataSet = ConvSingleCellDataset(X_val_img, y_val, img_size)
            TestDataSet = ConvSingleCellDataset(X_test_img, y_test, img_size)
            return_X_train = X_train_img
        elif model == "convnet":
            # Transform data into 1d images
            X_train_img = self.img_transform(X_train)
            X_val_img = self.img_transform(X_val)
            X_test_img = self.img_transform(X_test)

            # Create datasets
            img_size = X_train_img[0].shape[0]

            TrainDataSet = ConvSingleCellDataset(X_train_img, y_train, img_size)
            ValDataSet = ConvSingleCellDataset(X_val_img, y_val, img_size)
            TestDataSet = ConvSingleCellDataset(X_test_img, y_test, img_size)
            return_X_train = X_train_img
        elif model == "deepnet":
            # Create datasets
            TrainDataSet = DeepSingleCellDataset(X_train, y_train)
            ValDataSet = DeepSingleCellDataset(X_val, y_val)
            TestDataSet = DeepSingleCellDataset(X_test, y_test)
            return_X_train = X_train

        # Create dataloaders

        train_data_loader = DataLoader(TrainDataSet, batch_size=batch_size, shuffle=True)
        val_data_loader = DataLoader(ValDataSet, batch_size=batch_size, shuffle=True)
        test_data_loader = DataLoader(TestDataSet, batch_size=X_test.shape[0], shuffle=True)
        
        return train_data_loader, val_data_loader, test_data_loader, count_train, count_val, count_test, X, input_df, return_X_train