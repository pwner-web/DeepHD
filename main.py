import sys
import os
bin = os.path.abspath(os.path.dirname(__file__))
sys.path.append(bin + "/lib")
import glob
#import paramiko
import configparser
import datetime
import argparse
import logging
import pickle

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc

import ConvNetModel
import DeepNetModel
import ResNetModel
import prepare
import SHAP_analysis
import Plotshow

import warnings

__author__="Gao Shichen"
__mail__= "gaoshichend@163.com"
__date__= "2024/04/26"
__update__ = "2024/05/10"

__doc__ = """
This program is used to predict gene expression from single-cell datasets and includes three machine learning algorithms: 1. Deep Neural Network, 2. Convolutional Neural Network, 3. Deep Residual Neural Network.

Notes:

1.This is a binary classification model that supports only two labels.
2.The dataset should be a gene expression matrix of single-cell sequencing results, where cells are rows and genes are columns.
3.The training data is divided into upper and lower quartiles, but it can be modified according to the data characteristics as needed.
4.The original expression matrix should undergo preprocessing, including the following steps: removing bimodal genes, normalizing and scaling RNA counts, and removing genes with fewer than 200 counts across cells or expressed in fewer than 3 cells.
5.Default parameters: batch_size is 64, learning rate is 0.00005, and epochs is 30. These parameters can be fine-tuned based on the actual data size, number of genes, training results, loss values, etc. The parameters are stored in the config.ini file.
"""

# Configure logging and set the log level to INFO.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# function: call label based on preset probability cutoff
def call_label(pred, prob_cutoff):
    pred_label = []
    for i in pred.squeeze():
        if i >= prob_cutoff:
            pred_label.append(1)
        else:
            pred_label.append(0)
    return torch.tensor(pred_label).reshape(len(pred_label),)

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
logging.info("The program will run on the device: {device}.".format(device = device))

# Training loop
def train(dataloader, model, loss_fn, optimizer, scheduler, prob_cutoff, count):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct_high, correct_low, incorrect_high, incorrect_low  = 0, 0, 0, 0, 0
    high_count, low_count = count[0], count[1]
    
    true_labels = []
    predicted_scores = []

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.type(torch.float)
        y=torch.squeeze(y).type(torch.float)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(1))
        train_loss += loss.item()

        correct_high += (torch.logical_and(torch.round(pred.squeeze()) == 1, y == 1)).type(torch.float).sum().item()
        incorrect_high += (torch.logical_and(torch.round(pred.squeeze()) == 1, y == 0)).type(torch.float).sum().item()
        correct_low += (torch.logical_and(torch.round(pred.squeeze()) == 0, y == 0)).type(torch.float).sum().item()
        incorrect_low += (torch.logical_and(torch.round(pred.squeeze()) == 0, y == 1)).type(torch.float).sum().item()

        pred_probs = torch.sigmoid(pred).cpu().detach().numpy()
        true_labels.extend(y.cpu().numpy())
        predicted_scores.extend(pred_probs)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            logging.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    FPR, TPR, _ = roc_curve(true_labels, predicted_scores)
    roc_auc = auc(FPR, TPR)
    Precision = 100*correct_high/(correct_high+incorrect_high)
    Recall = 100*correct_high/(correct_high+incorrect_low)
    F1 = 2*Precision*Recall/(Precision+Recall)
    #TPR=correct_high/(correct_high+incorrect_low)
    #FPR=incorrect_high/(incorrect_high+correct_low)

    train_loss /= num_batches
    logging.info(f"Avg loss of train set: {train_loss:.4f} \n")
    logging.info(f"Correctly classified as high in train set: {correct_high}/{high_count} ({100*correct_high/high_count:>4f}%)")
    logging.info(f"Correctly classified as low in train set: {correct_low}/{low_count} ({100*correct_low/low_count:>4f}%)\n")
    logging.info(f"Precesion value in train set: ({Precision:>4f}%)\n")
    logging.info(f"Recall value in train set: ({Recall:>4f}%)\n")
    logging.info(f"F1 score in train set: ({F1:>4f}%)\n")
    logging.info(f"roc_auc in train set: ({roc_auc:>4f}%)\n")
    logging.info(f"Overall accuracy: {correct_high+correct_low}/{high_count+low_count} ({100*(correct_high+correct_low)/(high_count+low_count):.4f}%)\n")
    
    scheduler.step()
    
    return train_loss, 100*correct_high/high_count, 100*correct_low/low_count, F1, TPR, FPR, roc_auc

# Validation loop
def val(dataloader, model, loss_fn, prob_cutoff, count):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss, correct_high, correct_low, incorrect_high, incorrect_low = 0, 0, 0, 0, 0
    high_count, low_count = count[0], count[1]

    true_labels = []
    predicted_scores = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.type(torch.float)
            #linearInputLength = int(int(int(int(int(int(X.shape[2]/4)/2)/4)/2)/4)/2)*6
            #model.LinearInputLength = linearInputLength
            y=torch.squeeze(y).type(torch.float)
            
            pred = model(X)
            pred_label = call_label(pred, prob_cutoff).to(device)
            val_loss += loss_fn(pred, y.unsqueeze(1)).item()
            correct_high += (torch.logical_and(pred_label == 1, y == 1)).type(torch.float).sum().item()
            incorrect_high += (torch.logical_and(pred_label == 1, y == 0)).type(torch.float).sum().item()
            correct_low += (torch.logical_and(pred_label == 0, y == 0)).type(torch.float).sum().item()
            incorrect_low += (torch.logical_and(pred_label == 0, y == 1)).type(torch.float).sum().item()

            pred_probs = torch.sigmoid(pred).cpu().detach().numpy()
            true_labels.extend(y.cpu().numpy())
            predicted_scores.extend(pred_probs)
    
    FPR, TPR, _ = roc_curve(true_labels, predicted_scores)
    roc_auc = auc(FPR, TPR)
    Precision = 100*correct_high/(correct_high+incorrect_high)
    Recall = 100*correct_high/(correct_high+incorrect_low)
    F1 = 2*Precision*Recall/(Precision+Recall)
    #TPR=correct_high/(correct_high+incorrect_low)
    #FPR=incorrect_high/(incorrect_high+correct_low)

    val_loss /= num_batches
    logging.info(f"Avg loss of test set: {val_loss:.4f} \n")
    logging.info(f"Accuracy for 'high' class of validation set: {correct_high}/{high_count} ({100*correct_high/high_count:.4f}%)")
    logging.info(f"Accuracy for 'low' class of validation set: {correct_low}/{low_count} ({100*correct_low/low_count:.4f}%)")
    logging.info(f"Precesion value in validation set: ({Precision:>4f}%)\n")
    logging.info(f"Recall value in validation set: ({Recall:>4f}%)\n")
    logging.info(f"F1 score in validation set: ({F1:>4f}%)\n")
    logging.info(f"roc_auc in train set: ({roc_auc:>4f}%)\n")
    logging.info(f"Overall accuracy: {correct_high+correct_low}/{high_count+low_count} ({100*(correct_high+correct_low)/(high_count+low_count):.4f}%)\n")
    
    return val_loss, 100*correct_high/high_count, 100*correct_low/low_count, F1, TPR, FPR, roc_auc


def main():
    # Prepare the configuration file.
    configfile = configparser.ConfigParser()
    configfile.read(os.path.join(bin,"config","config.ini"))
    cut_off_high = float(configfile['dataset']['cut_off_high'])
    cut_off_low = float(configfile['dataset']['cut_off_low'])
    prob_cutoff = float(configfile['train']['prob_cutoff'])
    batch_size = int(configfile['train']['batch_size'])
    res_learning_rate = float(configfile['train']['res_learning_rate'])
    deep_learning_rate = float(configfile['train']['deep_learning_rate'])
    conv_learning_rate = float(configfile['train']['conv_learning_rate'])
    epochs = int(configfile['train']['epochs'])

    # Configure the parameters.
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="author:\t{0}\nmail:\t{1}\ndate:\t{2}\n".format(__author__,__mail__,__date__))
    parser.add_argument("-m", "--model", dest='model',choices=["resnet","convnet","deepnet"], type=str, default="resnet", help="Choose a neural network model: convnet、deepnet or resnet (Note: the default model is resnet)")
    parser.add_argument("-r", "--run", dest='run',choices=["train","explain"], type=str, default="train", help="Select the mode for training or explaining (Note: train is the default mode).")
    parser.add_argument("-t", "--target", dest='target',type=str, default="labels_list", help="Choose the target labels for prediction. (Note: 1.the default name is labels_list.)")
    parser.add_argument("--labels_high", dest='labels_high',type=str, help="Choose the labels high for prediction. (Note: 1.the label name will convert to the number 1")
    parser.add_argument("--labels_low", dest='labels_low',type=str, help="Choose the labels low for prediction. (Note: 1.the label name will convert to the number 0")
    parser.add_argument("-d", "--dataset", dest='dataset', type=str, default="{bindir}/data/HD_transformed_data.csv".format(bindir=bin), help="Choose a scRNA gene expression matrix dataset (the default dataset is example)")
    parser.add_argument("-s", "--state_dict", dest='state_dict', type=str, default="{bindir}/model/Model.pth".format(bindir=bin), help="Saving or loading PyTorch models for training, evaluation, or inference.")

    args = parser.parse_args()

    # Start training

    train_data_loader, val_data_loader, test_data_loader, count_train, count_val, count_test, X, input_df, X_train = prepare.PrepareprocessingDataset().data_structure(
            labels = args.target, 
            path = args.dataset, 
            model = args.model, 
            labels_high = args.labels_high, 
            labels_low = args.labels_low, 
            batch_size = batch_size
            )

    if args.run == "train":
        gene_number = X.shape[1]
        print("data shape: {shape}".format(shape=X.shape))
        print("gene_number:{gene_number}".format(gene_number=gene_number))

        #model = NeuralNetwork(input_size=gene_number).to(device)
        if args.model == "resnet":
            learning_rate = res_learning_rate
            linearInputLength = math.ceil(math.ceil(math.ceil((math.ceil((gene_number-1)/4)+1)/2-4)/2-4)/2+2)*6
            model = ResNetModel.NeuralNetwork(LinearInputLength=linearInputLength).to(device)
            print(model)
        elif args.model == "convnet":
            learning_rate = conv_learning_rate
            linearInputLength = int(int(int(int(int(int(gene_number/4)/2)/4)/2)/4)/2)*6
            model = ConvNetModel.NeuralNetwork(LinearInputLength=linearInputLength).to(device)
            print(model)
        elif args.model == "deepnet":
            learning_rate = deep_learning_rate
            model = DeepNetModel.NeuralNetwork(input_size=gene_number).to(device)
            print(model)

        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_loss = []
        train_accuracy_high = []
        train_accuracy_low = []
        train_F1 = []
        train_TPR = []
        train_FPR = []
        train_roc_auc = []

        val_loss = []
        val_accuracy_high = []
        val_accuracy_low = []
        val_F1 = []
        val_TPR = []
        val_FPR = []
        val_roc_auc = []

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            #train_data_loader, val_data_loader, test_data_loader, count_train, count_val, count_test, X, input_df, X_train = prepare.PrepareprocessingDataset().data_structure(
            #    labels = args.target, 
            #    path = args.dataset, 
            #    model = args.model, 
            #    labels_high = args.labels_high, 
            #    labels_low = args.labels_low, 
            #    batch_size = batch_size
            #)
            loss_train_epoch, acc_train_epoch_high, acc_train_epoch_low, F1, TPR, FPR, roc_auc = train(train_data_loader, model, loss_fn, optimizer, scheduler, prob_cutoff, count_train)
            train_loss.append(loss_train_epoch)
            train_accuracy_high.append(acc_train_epoch_high)
            train_accuracy_low.append(acc_train_epoch_low)
            train_F1.append(F1)
            train_TPR.append(TPR)
            train_FPR.append(FPR)
            train_roc_auc.append(roc_auc)

            loss_val_epoch, acc_val_epoch_high, acc_val_epoch_low, F1, TPR, FPR, roc_auc = val(val_data_loader, model, loss_fn, prob_cutoff, count_val)
            val_loss.append(loss_val_epoch)
            val_accuracy_high.append(acc_val_epoch_high)
            val_accuracy_low.append(acc_val_epoch_low)
            val_F1.append(F1)
            val_TPR.append(TPR)
            val_FPR.append(FPR)
            val_roc_auc.append(roc_auc)
            train_list = [train_loss, train_accuracy_high, train_accuracy_low, train_F1, train_TPR, train_FPR, train_roc_auc, val_loss, val_accuracy_high, val_accuracy_low, val_F1, val_TPR, val_FPR, val_roc_auc]
            file_path = '{path}/{model}_train_list.pkl'.format(path=bin + "/pickle",model=args.model)
            with open(file_path, 'wb') as file:
                pickle.dump(train_list, file)

        logging.info("Training finished.")

        torch.save(model, args.state_dict)
        logging.info("Model saved.")
        image = Plotshow.Image(epochs=epochs)
        image.lossfunc_plot(loss_fn=loss_fn, train_loss=train_loss, val_loss=val_loss, model_type=args.model, bindir=bin)
        image.accuracy_plot(train_accuracy_high=train_accuracy_high, train_accuracy_low=train_accuracy_low, val_accuracy_high=val_accuracy_high, val_accuracy_low=val_accuracy_low, model_type=args.model, bindir=bin)
        image.F1_score(train_F1=train_F1, val_F1=val_F1, model_type=args.model, bindir=bin)

    elif args.run == "finetune":
        gene_number = X.shape[1]
        model = DeepNetModel.NeuralNetwork(input_size=gene_number).to(device)
        print(args.state_dict)
        model.load_state_dict(torch.load(args.state_dict))
        model.train()

        for param in model.parameters():
            param.requires_grad = False
        for param in model.linear_relu_stack[:6].parameters():
            param.requires_grad = True

        loss_fn = nn.BCELoss()
        #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(model.linear_relu_stack[14:].parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_loss = []
        train_accuracy_high = []
        train_accuracy_low = []

        val_loss = []
        val_accuracy_high = []
        val_accuracy_low = []

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            loss_train_epoch, acc_train_epoch_high, acc_train_epoch_low = train(train_data_loader, model, loss_fn, optimizer, scheduler, prob_cutoff, count_train)
            train_loss.append(loss_train_epoch)
            train_accuracy_high.append(acc_train_epoch_high)
            train_accuracy_low.append(acc_train_epoch_low)

            loss_val_epoch, acc_val_epoch_high, acc_val_epoch_low = val(val_data_loader, model, loss_fn, prob_cutoff, count_val)
            val_loss.append(loss_val_epoch)
            val_accuracy_high.append(acc_val_epoch_high)
            val_accuracy_low.append(acc_val_epoch_low)

        logging.info("Training finished.")

        torch.save(model, "finetune_Model.pth")
        logging.info("Model saved.")
        image = Plotshow.Image(epochs=epochs)
        image.lossfunc_plot(loss_fn=loss_fn, train_loss=train_loss, val_loss=val_loss)
        image.accuracy_plot(train_accuracy_high=train_accuracy_high, train_accuracy_low=train_accuracy_low, val_accuracy_high=val_accuracy_high, val_accuracy_low=val_accuracy_low)


    elif args.run == "explain":
        model_eval = torch.load(args.state_dict)
        logging.info("Model loaded.")

        gene_list = input_df.columns[1:-1]

        rng = np.random.default_rng(1111)

        random_sample_num = 2000
        random_index = rng.choice(range(X_train.shape[0]), random_sample_num, replace = False, shuffle = False)
        random_train_data = torch.tensor(X[random_index]).float().to(device)
        if args.model != "deepnet":
            # add channels
            random_train_data = torch.unsqueeze(random_train_data, dim=1)

        explainer_training, shap_values_training, shap_values_summary, data, feature_list = SHAP_analysis.shap_explainer(model_eval, random_train_data, gene_list)
        shaply_list = [explainer_training, shap_values_training, shap_values_summary, data, feature_list]
        file_path = '{path}/{model}_shaply_list.pkl'.format(path=bin + "/pickle",model=args.model)
        with open(file_path, 'wb') as file:
            pickle.dump(shaply_list, file)
        logging.info("Model explained")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
