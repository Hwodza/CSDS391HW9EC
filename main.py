import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import time


# Function to create and train neural networks
def create_and_train_nn(hidden_layers, X_train, y_train, X_test, y_test, max_iter=1000):
    # Define activation functions
    activations = ['logistic', 'tanh', 'relu']
    models = []
    accuracies = []
    runtimes = []
    
    for activation in activations:
        model = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, solver='adam', warm_start=True, max_iter=1)
        start_time = time.time()
        accuracy = []
        for _ in range(max_iter):
            model.partial_fit(X_train, y_train, classes=np.unique(y_train))
            accuracy.append(model.score(X_test, y_test))  # Test accuracy
        runtime = time.time() - start_time
        accuracies.append((activation, hidden_layers, accuracy))
        runtimes.append((activation, hidden_layers, runtime))
        models.append((activation, hidden_layers, model))
    
    return models, accuracies, runtimes


def main():
    # Load the data
    data = pd.read_csv('irisdata.csv')

    # Encode the species column to numerical values
    label_encoder = LabelEncoder()
    data['species'] = label_encoder.fit_transform(data['species'])

    # Split the data into features and target
    X = data.drop('species', axis=1)
    y = data['species']

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    nn_models_1_layer_3_nodes, accuracies_1_layer_3_nodes, runtimes_1_layer_3_nodes = create_and_train_nn((3), X_train, y_train, X_test, y_test)

    plt.figure(figsize=(10, 5))
    for activation, hidden_layers, accuracy in accuracies_1_layer_3_nodes:
        plt.plot(accuracy, label=f'{activation} {hidden_layers}')
    plt.xlabel('Iteration')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs Iteration')
    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    main()