# clustering_lab_exercise.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# === Load Dataset ===
# path to dataset
df = pd.read_csv("Mall_Customers.csv")

# === Preprocess: Select Numeric Features ===
# We'll use 'Annual Income' and 'Spending Score'
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize data for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Part 1: Elbow Method to Determine Optimal k ===
sse = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

# Plot SSE vs k
plt.figure(figsize=(8, 5))
plt.plot(k_range, sse, marker='o')
plt.title("Elbow Method - Dikshith Reddy M - Student 0789055")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_plot.png")
plt.show()

'''Based on the Elbow Method plot, the optimal number of clusters is 5.
The plot shows a sharp decrease in SSE until k=5, after which the improvement levels off.
This suggests that 5 clusters balance accuracy and simplicity, avoiding underfitting (too few clusters) or overfitting (too many clusters).
Using 5 clusters effectively separates customers by their income and spending patterns, helping identify distinct customer segments for marketing or analysis.'''

# === Part 2: Additional Clustering Methods ===

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
db_labels = dbscan.fit_predict(X_scaled)
print("DBSCAN Cluster Labels:", set(db_labels))

# Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=5)
agglo_labels = agglo.fit_predict(X_scaled)
print("Agglomerative Clustering Labels:", set(agglo_labels))

# === Part 3: Bonus - Suboptimal Cluster Counts ===

# Too Few Clusters (e.g., k=2)
kmeans_few = KMeans(n_clusters=2, random_state=42)
labels_few = kmeans_few.fit_predict(X_scaled)

# Too Many Clusters (e.g., k=8)
kmeans_many = KMeans(n_clusters=8, random_state=42)
labels_many = kmeans_many.fit_predict(X_scaled)

# Plot both side by side
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_few, cmap='viridis')
ax[0].set_title("Too Few Clusters (k=2)")
ax[0].set_xlabel("Annual Income (scaled)")
ax[0].set_ylabel("Spending Score (scaled)")

ax[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_many, cmap='viridis')
ax[1].set_title("Too Many Clusters (k=8)")
ax[1].set_xlabel("Annual Income (scaled)")
ax[1].set_ylabel("Spending Score (scaled)")

plt.tight_layout()
plt.savefig("too_few_and_many_clusters.png")
plt.show()


'''When using too few clusters (k=2), distinct groups in the data are merged, leading to oversimplified segmentation.
On the other hand, too many clusters (k=8) over-segment the data, splitting natural groups into meaningless sub-clusters.
The plots demonstrate that poor choice of k leads to loss of interpretability and analytical value.'''
