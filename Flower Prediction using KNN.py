import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# Flower labels and setup
flower_types = ['rose', 'daisy', 'lotus', 'sunflower', 'jasmine', 'lily']
num_samples_per_class = 20
np.random.seed(42)

# Generate synthetic flower features
def generate_flower_features(flower_name):
    if flower_name == 'rose':
        return np.random.normal([5.5, 2.0, 8, 200], [0.5, 0.2, 1.0, 15], (num_samples_per_class, 4))
    elif flower_name == 'daisy':
        return np.random.normal([3.0, 1.0, 4, 180], [0.4, 0.2, 1.0, 10], (num_samples_per_class, 4))
    elif flower_name == 'lotus':
        return np.random.normal([6.0, 2.5, 7, 150], [0.5, 0.3, 0.8, 12], (num_samples_per_class, 4))
    elif flower_name == 'sunflower':
        return np.random.normal([7.5, 3.0, 3, 230], [0.6, 0.4, 1.0, 10], (num_samples_per_class, 4))
    elif flower_name == 'jasmine':
        return np.random.normal([4.0, 1.5, 9, 255], [0.3, 0.2, 1.0, 5], (num_samples_per_class, 4))
    elif flower_name == 'lily':
        return np.random.normal([5.0, 2.0, 6, 170], [0.4, 0.3, 1.2, 10], (num_samples_per_class, 4))

# Create dataset
X = []
y = []
for idx, flower in enumerate(flower_types):
    features = generate_flower_features(flower)
    X.append(features)
    y.extend([idx] * num_samples_per_class)

X = np.vstack(X)
y = np.array(y)

# Train classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# Get user input for prediction
print("🌼 Enter your flower's features:")
petal_length = float(input("Petal Length (cm): "))
petal_width = float(input("Petal Width (cm): "))
fragrance = float(input("Fragrance Intensity (0–10): "))
color_hue = float(input("Color Hue (0–255): "))
sample = [[petal_length, petal_width, fragrance, color_hue]]

# Predict flower type
prediction = knn.predict(sample)[0]
print(f"\n🌸 The predicted flower type is **{flower_types[prediction]}**.")

# Calculate and display accuracy
train_pred = knn.predict(X)
accuracy = accuracy_score(y, train_pred)
print(f"✅ Training Accuracy: {accuracy * 100:.2f}%")

# Scatter plot of petal length vs. petal width
plt.figure(figsize=(8, 6))
colors = ListedColormap(['red', 'orange', 'green', 'blue', 'purple', 'pink'])

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colors, edgecolor='k', alpha=0.6)
plt.scatter(petal_length, petal_width, color='black', marker='x', s=100, label='Your Sample')
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("🌺 Flower Classification Scatter Plot")
plt.legend()
plt.grid(True)
plt.show()
