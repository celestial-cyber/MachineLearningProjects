# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Load the dataset
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names

# Convert to DataFrame
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

print("🔍 First 5 rows of the original dataset:")
print(df.head())
# Output:
#    alcohol  malic_acid  ash  ...  proline  target
# 0    14.23        1.71  2.43  ...   1065.0       0
# 1    13.20        1.78  2.14  ...   1050.0       0
# 2    13.16        2.36  2.67  ...   1185.0       0
# 3    14.37        1.95  2.50  ...   1480.0       0
# 4    13.24        2.59  2.87  ...    735.0       0

print("\n📊 Dataset shape:", X.shape)
# Output: (178, 13)

print("🧪 Number of classes:", len(np.unique(y)))
# Output: 3

print("🧬 Feature names:", feature_names)
# Output: ['alcohol', 'malic_acid', ..., 'proline']

# Step 2: Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n📏 After standardization:")
print("Mean (approx 0):", np.round(X_scaled.mean(axis=0), 2))
print("Std Dev (approx 1):", np.round(X_scaled.std(axis=0), 2))
# Output:
# Mean: [0. 0. 0. ... 0.]
# Std Dev: [1. 1. 1. ... 1.]

# Step 3: Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\n📉 Shape after PCA transformation:", X_pca.shape)
# Output: (178, 2)

# Step 4: Explained variance
explained_variance = pca.explained_variance_ratio_
print("\n📈 Explained Variance Ratio:")
for i, ratio in enumerate(explained_variance):
    print(f"PC{i+1}: {ratio:.4f}")
# Output:
# PC1: 0.3619
# PC2: 0.1920

print("\n📊 Total Variance Captured:", explained_variance.sum())
# Output: 0.5539

# Step 5: PCA Components
print("\n🧭 PCA Components (Eigenvectors):")
print(np.round(pca.components_, 4))
# Output:
# [[ 0.1443 -0.2452  0.0027 ...  0.2866]
#  [-0.4836 -0.2240 -0.0890 ...  0.1308]]

# Step 6: Projected Data
print("\n📦 First 5 samples of PCA-transformed data:")
print(np.round(X_pca[:5], 4))
# Output:
# [[2.1930 0.9880]
#  [2.5090 0.9990]
#  [2.7320 1.4080]
#  [2.8220 0.9090]
#  [1.4150 2.7750]]

# Create DataFrame for PCA results
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['target'] = y

print("\n🗂 First 5 rows of PCA DataFrame:")
print(df_pca.head())
# Output:
#     PC1    PC2  target
# 0  2.193  0.988       0
# 1  2.509  0.999       0
# 2  2.732  1.408       0
# 3  2.822  0.909       0
# 4  1.415  2.775       0

# Step 7: Visualize PCA result
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']
for i, label in enumerate(np.unique(y)):
    plt.scatter(df_pca[df_pca['target'] == label]['PC1'],
                df_pca[df_pca['target'] == label]['PC2'],
                color=colors[i], label=f"Class {label}", alpha=0.6)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Wine Dataset PCA Projection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Output: A scatter plot showing three distinct clusters for wine classes
