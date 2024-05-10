import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

__author__="Gao Shichen"
__mail__= "gaoshichend@163.com"
__date__= "2024/04/26"
__update__ = "2024/05/10"

class Image():
    def __init__(self, epochs):
        self.epochs = epochs

    def lossfunc_plot(self, loss_fn, train_loss, val_loss, model_type, bindir):
        # Plot training set and validation set loss
        fig = plt.figure()
        plt.figure(fig.number)  # Use separate figure object
        plt.plot(np.arange(1, self.epochs+1), train_loss, label="train_loss")
        plt.plot(np.arange(1, self.epochs+1), val_loss, label="test_loss")
        plt.xlabel('Epoch')
        plt.ylabel(str(loss_fn))
        plt.legend(loc=1)
        plt.xticks(np.arange(0, self.epochs+2, step=2))
        plt.xlim(1, self.epochs)
        plt.grid()
        plt.savefig(os.path.join(bindir,'result-tmp','{model_type}_loss_func.png'.format(model_type=model_type)))
        #plt.show()

    def accuracy_plot(self, train_accuracy_high, train_accuracy_low, val_accuracy_high, val_accuracy_low, model_type, bindir):
        # Plot training set and validation set accuracy
        fig = plt.figure()
        plt.figure(fig.number)  # Use separate figure object
        plt.plot(np.arange(1, self.epochs+1), train_accuracy_high, label="train_accuracy_high")
        plt.plot(np.arange(1, self.epochs+1), train_accuracy_low, label="train_accuracy_low")
        plt.plot(np.arange(1, self.epochs+1), val_accuracy_high, label="val_accuracy_high")
        plt.plot(np.arange(1, self.epochs+1), val_accuracy_low, label="val_accuracy_low")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy(%)')
        plt.legend(loc=4)
        plt.xticks(np.arange(0, self.epochs+2, step=2))
        plt.yticks(np.arange(0, 110, step=10))
        plt.xlim(1, self.epochs)
        plt.ylim(0,105)
        plt.grid()
        plt.savefig(os.path.join(bindir,'result-tmp','{model_type}_accuracy.png'.format(model_type=model_type)))
        #plt.show()
    
    def F1_score(self, train_F1, val_F1, model_type, bindir):
        # Plot training set and validation set F1 scores
        fig = plt.figure()
        plt.figure(fig.number)  # Use separate figure object
        plt.plot(np.arange(1, self.epochs+1), train_F1, label="train_F1_score")
        plt.plot(np.arange(1, self.epochs+1), val_F1, label="val_F1_score")
        plt.xlabel('Epoch')
        plt.ylabel('F1 score')
        plt.legend(loc=1)
        plt.xticks(np.arange(0, self.epochs+2, step=2))
        plt.xlim(1, self.epochs)
        plt.grid()
        plt.savefig(os.path.join(bindir,'result-tmp','{model_type}_F1_score.png'.format(model_type=model_type)))

def main(): pass

if __name__ == '__main__':
    main()
