# DeepHD
Deep residual network for Huntington's disease
# Gene Expression Prediction from Single-cell Datasets

This repository contains a Python program designed to predict gene expression from single-cell datasets using three different machine learning algorithms: Deep Neural Network, Convolutional Neural Network, and Deep Residual Neural Network.

## Notes:

1. This program supports binary classification models that work with only two labels.
2. The dataset should be provided as a gene expression matrix of single-cell sequencing results, where cells represent rows and genes represent columns.
3. The training data is divided into upper and lower quartiles, but this division can be adjusted based on the characteristics of the data.
4. Preprocessing of the original expression matrix is required, including steps such as removing bimodal genes, normalizing and scaling RNA counts, and filtering out genes with fewer than 200 counts across cells or expressed in fewer than 3 cells.
5. Default parameters are set as follows: batch_size is 64, learning rate is 0.001, and epochs is 30. However, these parameters can be fine-tuned based on factors such as data size, number of genes, training results, and loss values. Parameter configurations are stored in the config.ini file.

## Dependencies:

- Python 3
- PyTorch
- NumPy
- pandas
- Matplotlib
- scikit-learn
- tqdm

## Usage:

- Ensure all dependencies are installed.
- Execute the `main.py` script with appropriate arguments to train the model, perform model explanation, or fine-tune a pre-trained model.
- Example usage: 
    ```
    python main.py -m resnet -r train -t labels_list -d data/HD_transformed_data.csv
    ```
- For more details on available options and arguments, refer to the help text provided within the script.

## Author Information:

- **Author**: Gao Shichen
- **Email**: gaoshichend@163.com
- **Date**: 2024/04/23
- **Last Update**: 2023/05/11
