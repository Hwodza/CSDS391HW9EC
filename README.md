# CSDS391HW9EC
This project analyzes the Iris dataset using neural networks with different activation functions and architectures. The goal is to compare the performance of various neural network configurations.

## Requirements
- Python 3.x
- pandas
- scikit-learn
- matplotlib
- numpy

You can install the required packages using pip:
```sh
pip install pandas scikit-learn matplotlib numpy
```
## Files
- **main.py:** The main script that performs the analysis.
- **irisdata.csv** The csv file with the irisdata

## Usage
1. Ensure that the irisdata.csv file is in the same directory as main.py.
2. Run the main.py script:
```sh
python main.py
```
## Description
### Data Preprocessing
1. The Iris dataset is loaded and the species column is encoded to numerical values.
2. The features are standardized using StandardScaler.
3. The data is split into training and testing sets.
### Neural Network Training
1. Neural networks with different architectures and activation functions (logistic, tanh, relu) are trained.
2. Each neural network configuration is trained 10 times with different random states.
3. The mean and variance of the accuracies and runtimes are calculated.
### Comparison
1. The final accuracies for each trial of the neural network training are printed.
2. The mean and variance of the accuracies and runtimes are printed.
3. Bar plots are generated to visualize the mean and variance of the accuracies and runtimes for each neural network configuration.
### Output
The script prints the following information:

- Final accuracies for each trial of the neural network training.
- Mean and variance of the accuracies and runtimes for each neural network configuration.
  
#### Bar plots are displayed showing:

- Mean accuracies for each neural network configuration.
- Variance of accuracies for each neural network configuration.
- Mean runtimes for each neural network configuration.
- Variance of runtimes for each neural network configuration.