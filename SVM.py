# Install necessary libraries before running this code (only once):
# pip install scikit-learn matplotlib

# Import necessary libraries
import numpy as np                      # For numerical operations and array handling
import matplotlib.pyplot as plt         # For visualization
from sklearn import datasets            # To load built-in datasets like Iris
from sklearn.model_selection import train_test_split  # For splitting dataset into training and testing
from sklearn.svm import SVC             # SVC = Support Vector Classifier (SVM algorithm)
from sklearn.metrics import classification_report, confusion_matrix  # For evaluating the model

# --------------------------------------
# STEP 1: Load the Iris dataset
# --------------------------------------
iris = datasets.load_iris()            # Load the Iris flower dataset
X = iris.data[:, :2]                   # Take only the first two features: Sepal length & Sepal width (for 2D visualization)
y = iris.target                        # Target labels (0 = setosa, 1 = versicolor, 2 = virginica)

# --------------------------------------
# STEP 2: Split the dataset
# --------------------------------------
# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------
# STEP 3: Create and train the SVM model
# --------------------------------------
model = SVC(kernel='linear')           # Create a Support Vector Machine classifier with a linear kernel
model.fit(X_train, y_train)            # Train the model using training data

# --------------------------------------
# STEP 4: Make predictions and evaluate
# --------------------------------------
y_pred = model.predict(X_test)         # Predict labels on the test set

# Print confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))    # Shows how many predictions were correct or incorrect

print("\nClassification Report:")
print(classification_report(y_test, y_pred))  # Shows precision, recall, f1-score, accuracy, etc.

# --------------------------------------
# STEP 5: Visualize Decision Boundaries
# --------------------------------------

# Define the range of x and y for the plot
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1   # Sepal length range
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1   # Sepal width range

# Create a grid of points covering the plot area
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.01),
    np.arange(y_min, y_max, 0.01)
)

# Predict the label for each point in the grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)   # Reshape to match the grid shape

# Draw the decision boundary using contour plot
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)  # Fill decision areas with colors

# Scatter plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', label='Train', edgecolors='k')

# Scatter plot the testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='s', label='Test', edgecolors='k')

# Add labels and title
plt.title('SVM Decision Boundary (Iris Dataset - Sepal Features)')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.grid(True)
plt.show()
