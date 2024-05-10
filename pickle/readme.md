# Pickle Directory

The `pickle` directory within this repository serves as a storage location for serialized Python objects using the Pickle module. These serialized objects include intermediate results, model outputs, and other data structures generated during the execution of the main program.

## Contents:

- **[model_name]_train_list.pkl**: This file contains lists storing various metrics and performance indicators during the training process of the specified model. Metrics such as training loss, accuracy, F1 score, and ROC-AUC values are included.

- **[model_name]_shaply_list.pkl**: This file contains SHAP (SHapley Additive exPlanations) values and related data structures generated during the explanation phase of the specified model. SHAP values are used to explain individual predictions made by the model and provide insights into feature importance.

## Usage:

These serialized objects can be loaded and utilized for further analysis, visualization, or model interpretation. The contents of each Pickle file can be accessed programmatically using Python's Pickle module.
