import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Step 1: Load the heart disease dataset
# The dataset contains patient features with a target column indicating heart disease presence (1 or 0)
df = pd.read_csv("C:/Users/DELL/Desktop/MLIIIyear/MLlab/Datasets/heartdiseasedataset.csv")

# Output: Display the first few rows for an overview of the data structure
print("Dataset preview:")
print(df.head())

# Step 2: Understand data structure and types
print("\nDataset info:")
print(df.info())

# Step 3: Separate features from the target variable
X = df.drop("target", axis=1)  # Features
y = df["target"]               # Labels: heart disease presence (1) or absence (0)

# Step 4: Apply the EM algorithm via Gaussian Mixture Model for clustering into 2 groups
gmm = GaussianMixture(n_components=2, random_state=42)
gmm_labels = gmm.fit_predict(X)

# Step 5: Apply KMeans clustering into 2 groups
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

# Step 6: Evaluate clustering performance compared to actual target labels using Adjusted Rand Index (ARI)
gmm_ari = adjusted_rand_score(y, gmm_labels)
kmeans_ari = adjusted_rand_score(y, kmeans_labels)

# Output: Display ARI scores which reflect how well the clusters correspond to actual heart disease classification
print("\nAdjusted Rand Index (ARI) Scores:")
print(f"GMM EM Algorithm: {gmm_ari:.4f}")  # Expected ~0.06, low correlation
print(f"KMeans: {kmeans_ari:.4f}")         # Expected ~0.02, very low correlation
