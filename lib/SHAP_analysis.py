import pandas as pd
import numpy as np
import os
import sys 
import torch
from torch import nn

from sklearn.model_selection import train_test_split

import shap
import matplotlib.pyplot as plt
from IPython.display import display

__author__="Gao Shichen"
__mail__= "gaoshichend@163.com"
__date__= "2023/05/26"
__update__ = "2023/07/12"

def blockprint(func):
    def wrapper(*args, **kwargs):
        sys.stdout = open(os.devnull, 'w')
        results = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return results
    return wrapper

# Define SHAP analysis function
@blockprint
def shap_explainer(model, data, feature_list):
    shap.initjs()
    explainer = shap.DeepExplainer(model, data)
    shap_values = explainer.shap_values(data)
    shap_values = np.squeeze(shap_values)
    # numpy squeeze [1000, 1, 17500]
    shap_values = np.squeeze(shap_values)
    # torch tensor squeeze torch.tensor([1000, 1, 17500])
    data = data.squeeze(dim=1)
    #p = shap.force_plot(base_value=explainer.expected_value, shap_values=shap_values[1][:20], features=feature_list[:20], show=True, link="logit")
    #display(p)
    #shap.decision_plot(base_value=explainer.expected_value, shap_values=shap_values[1][:20], features=feature_list[:20])
    # summary plot
    #shap.summary_plot(shap_values, feature=data,feature_names=feature_list, max_display=25, show=True, plot_size=(10,15))
    #shap.summary_plot(shap_values, features=data, feature_names=feature_list, max_display=25, plot_type="dot", alpha=1, show=True, plot_size=(10,15))
    #shap.summary_plot(shap_values, features=data, feature_names=feature_list, max_display=10, plot_type="bar", alpha=1, show=True, plot_size=(10,15))
    
    # save basic stats of shap values
    shap_values_df = pd.DataFrame(shap_values)
    shap_values_summary = pd.DataFrame(np.abs(shap_values_df).mean(), columns=['Mean Absolute SHAP value'])
    shap_values_summary['Max Absolute SHAP value'] = np.abs(shap_values_df).max()
    shap_values_summary['Median Absolute SHAP value'] = np.abs(shap_values_df).median()
    shap_values_summary["Gene"] = feature_list

    return explainer, shap_values, shap_values_summary, data, feature_list

if __name__ == '__main__':
    # SHAP analysis
    gene_list = input_df.columns[1:-1]

    rng = np.random.default_rng(1111)
    random_sample_num = 1000
    random_index = rng.choice(range(X_train.shape[0]), random_sample_num, replace = False, shuffle = False)
    random_train_data = torch.tensor(X[random_index]).float()

    explainer_training, shap_values_training, shap_values_summary, data, feature_list = shap_explainer(model_eval, random_train_data, gene_list)