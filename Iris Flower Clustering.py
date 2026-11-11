
# =====================================================
# MINI PROJECT 3: IRIS FLOWER CLUSTERING PROJECT
# Algorithm: K-Means Clustering
# Dataset: Iris Dataset (UCI / Scikit-learn)
# =====================================================

# 1. Import Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

# 2. Load Iris Dataset

iris = load_iris()

X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df["True Label"] = y

print("First Five Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# 3. Data Preprocessing

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply K-Means Clustering (k=3)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

predicted_clusters = kmeans.fit_predict(X_scaled)

df["Predicted Cluster"] = predicted_clusters

print("\nK-Means Clustering Completed!")
print("\nCluster Counts:")
print(df["Predicted Cluster"].value_counts().sort_index())

# 5. Visualize Clusters Using Scatter Plot

plt.figure(figsize=(9, 6))

plt.scatter(
    X_scaled[:, 2],
    X_scaled[:, 3],
    c=predicted_clusters,
    cmap="viridis",
    s=60
)

plt.xlabel("Petal Length (Standardized)")
plt.ylabel("Petal Width (Standardized)")
plt.title("Iris Flower Clusters using K-Means")
plt.colorbar(label="Cluster")

plt.tight_layout()
plt.show()

# 6. Reduce Dimensions using PCA for Visualization

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(9, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=predicted_clusters,
    cmap="viridis",
    s=60
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering Visualization using PCA")
plt.colorbar(label="Predicted Cluster")

plt.tight_layout()
plt.show()

# 7. Compare Predicted Clusters with True Labels

print("\nTrue Labels:")
print(iris.target_names)

print("\nConfusion Matrix (True Labels vs Predicted Clusters):")

print(confusion_matrix(y, predicted_clusters))

# 8. Visualize True Labels vs Predicted Clusters

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="viridis",
    s=60
)

axes[0].set_title("Actual Iris Flower Labels")
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")

axes[1].scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=predicted_clusters,
    cmap="viridis",
    s=60
)

axes[1].set_title("Predicted K-Means Clusters")
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")

plt.tight_layout()
plt.show()

# 9. Final Project Summary

print("\n========== PROJECT SUMMARY ==========")

print("Project: Iris Flower Clustering")
print("Algorithm: K-Means Clustering")
print("Number of Clusters: 3")
print("Total Samples:", len(X))
print("Number of Features:", X.shape[1])

print("\nProject Completed Successfully!")

