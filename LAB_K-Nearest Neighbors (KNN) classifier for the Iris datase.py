# Improved and Commented Version of K-Nearest Neighbors (KNN) Classifier for Iris Dataset

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets, neighbors
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data  # Feature matrix
y = iris.target  # Target labels

print("Keys in iris dataset:", iris.keys())
print("Shape of data (samples, features):", X.shape)
print("First data sample:", X[0])
print("Shape of target:", y.shape)
print("Target labels:", y)
print("Target names:", iris.target_names)

# Visualize 2D scatter plot for first two features
x_index = 0  # sepal length
y_index = 1  # sepal width

# Colorbar formatter to map target values to class names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

plt.figure(figsize=(8, 6))
plt.scatter(X[:, x_index], X[:, y_index], c=y, cmap=plt.cm.get_cmap('RdYlBu', 3))
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.clim(-0.5, 2.5)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.title("Iris Dataset - First Two Features")
plt.show()

# Train a KNN classifier
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# Predict a sample
sample = [[3, 5, 4, 2]]
prediction = knn.predict(sample)
print("Predicted class for sample", sample, ":", iris.target_names[prediction][0])

# Print class probabilities for the sample
probs = knn.predict_proba(sample)
print("Class probabilities:", probs)

# Plot decision boundaries using only the first two features
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_iris_knn():
    X_plot = X[:, :2]  # Using first two features for plotting
    y_plot = y

    knn_2d = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn_2d.fit(X_plot, y_plot)

    # Create a mesh grid
    x_min, x_max = X_plot[:, 0].min() - 0.1, X_plot[:, 0].max() + 0.1
    y_min, y_max = X_plot[:, 1].min() - 0.1, X_plot[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Predict on mesh grid
    Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

    # Plot training points
    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, cmap=cmap_bold, edgecolor='k')
    plt.xlabel('Sepal length (cm)')
    plt.ylabel('Sepal width (cm)')
    plt.title('KNN (k=3) Decision Boundaries on Iris Dataset (2D)')
    plt.axis('tight')
    plt.show()

plot_iris_knn()

# Evaluate model predictions
predictions = knn.predict(X)
correct = np.sum(predictions == y)
wrong = np.sum(predictions != y)
print(f"Correct predictions: {correct}")
print(f"Wrong predictions: {wrong}")
print(f"Accuracy: {accuracy_score(y, predictions)*100:.2f}%")
