import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import time
import random


# random.seed(0)


def create_and_train_nn(hidden_layers, X_train, y_train, X_test, y_test, max_iter=1000, random_state=1):
    activations = ['logistic', 'tanh', 'relu']
    models = []
    accuracies = []
    runtimes = []
    
    for activation in activations:
        model = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, solver='adam', warm_start=True, max_iter=1, random_state=random_state)
        start_time = time.time()
        accuracy = []
        for _ in range(max_iter):
            model.partial_fit(X_train, y_train, classes=np.unique(y_train))
            accuracy.append(model.score(X_test, y_test))  
        runtime = time.time() - start_time
        accuracies.append((activation, hidden_layers, accuracy))
        runtimes.append((activation, hidden_layers, runtime))
        models.append((activation, hidden_layers, model))
    
    return models, accuracies, runtimes


def repeat_nn(hidden_layers, X_train, y_train, X_test, y_test, max_iter=1000, n_repeats=10):
    accuracies = []
    runtimes = []
    models = []
    for _ in range(n_repeats):
        random_state = random.randint(0, 1000)
        nn_models, nn_accuracies, nn_runtimes = create_and_train_nn(hidden_layers, X_train, y_train, X_test, y_test, max_iter, random_state)
        accuracies.append(nn_accuracies)
        runtimes.append(nn_runtimes)
        models.append(nn_models)
    return models, accuracies, runtimes


def plot_repeat_nn(nn_models, accuracies, runtimes):
    # Print final accuracies for all 10 trials
    for i, accs in enumerate(accuracies):
        print(f"Trial {i+1} Final Accuracies:")
        for activation, hidden_layers, accuracy in accs:
            print(f"Activation: {activation}, Hidden Layers: {hidden_layers}, Final Accuracy: {accuracy[-1]:.2f}")

    # Calculate mean and variance of accuracies and runtimes
    mean_accuracies = np.mean([np.array([acc[-1] for _, _, acc in accs]) for accs in accuracies], axis=0)
    var_accuracies = np.var([np.array([acc[-1] for _, _, acc in accs]) for accs in accuracies], axis=0)
    mean_runtimes = np.mean([np.array([rt for _, _, rt in rts]) for rts in runtimes], axis=0)
    var_runtimes = np.var([np.array([rt for _, _, rt in rts]) for rts in runtimes], axis=0)
    
    # Print mean and variance of accuracies and runtimes
    print("Mean Accuracies:", mean_accuracies)
    print("Variance of Accuracies:", var_accuracies)
    print("Mean Runtimes:", mean_runtimes)
    print("Variance of Runtimes:", var_runtimes)

    labels = [f'{activation} {hidden_layers}' for activation, hidden_layers, _ in accuracies[0]]

    # Plot mean accuracies
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, mean_accuracies)
    plt.xlabel('Model')
    plt.ylabel('Mean Accuracy')
    plt.title('Mean Accuracies')
    # plt.xticks(rotation=90)
    for bar, mean in zip(bars, mean_accuracies):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{mean}', ha='center', va='bottom')
    plt.show()

    # Plot var accuracies
    plt.figure(figsize=(10, 5))
    labels = [f'{activation} {hidden_layers}' for activation, hidden_layers, _ in accuracies[0]]
    bars = plt.bar(labels, var_accuracies)
    plt.xlabel('Model')
    plt.ylabel('Variance of Accuracy')
    plt.title('Variance of Accuracies')
    # plt.xticks(rotation=90)
    for bar, mean in zip(bars, var_accuracies):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{mean}', ha='center', va='bottom')
    plt.show()

    # Plot mean runtimes
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, mean_runtimes)
    plt.xlabel('Model')
    plt.ylabel('Mean Runtime (s)')
    plt.title('Mean Runtimes')
    # plt.xticks(rotation=90)
    for bar, mean in zip(bars, mean_runtimes):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{mean}', ha='center', va='bottom')
    plt.show()

    # Plot var runtimes
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, var_runtimes)
    plt.xlabel('Model')
    plt.ylabel('Var Runtime (s)')
    plt.title('Var Runtimes')
    # plt.xticks(rotation=90)
    for bar, mean in zip(bars, var_runtimes):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{mean}', ha='center', va='bottom')
    plt.show()


def plot_single(nn_models, accuracies, runtimes):
    plt.figure(figsize=(10, 5))
    for activation, hidden_layers, accuracy in accuracies:
        plt.plot(accuracy, label=f'{activation} {hidden_layers}')
        plt.annotate(f'{accuracy[-1]:.2f}', xy=(len(accuracy)-1, accuracy[-1]), textcoords='offset points', xytext=(0,5), ha='center')
        print(f'Activation: {activation}\nNumber of Hidden Layers: {hidden_layers}\nFinal Test Accuracy: {accuracy[-1]:.2f}\n')
    plt.xlabel('Iteration')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs Iteration')
    plt.legend()
    plt.show()


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



    # Create and train 3 nn models with 1 hidden layer and 3 nodes, one for each activation function (logistic, tanh, relu)
    nn_models_1_layer_3_nodes, accuracies_1_layer_3_nodes, runtimes_1_layer_3_nodes = create_and_train_nn((3), X_train, y_train, X_test, y_test)
    plot_single(nn_models_1_layer_3_nodes, accuracies_1_layer_3_nodes, runtimes_1_layer_3_nodes)

    
    nn = [(3), (10), (10, 10), (10, 10, 10), (10, 10, 10, 10)]
    nn = [(10, 10, 10)]
    for hidden_layers in nn:
        nn_models, accuracies, runtimes = repeat_nn(hidden_layers, X_train, y_train, X_test, y_test)
        plot_repeat_nn(nn_models, accuracies, runtimes)


if __name__ == "__main__":
    main()