# ============================================================
# BEST SELLING PRODUCT IDENTIFICATION SYSTEM
# Professional ML Pipeline Project (Final Version)
# ============================================================

# =============================
# Import Libraries
# =============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage

import warnings
warnings.filterwarnings('ignore')


# =============================
# Plot Function (Safe PCA)
# =============================

def plot_clusters(X_pca, labels, title):

    plt.figure(figsize=(8,6))

    if X_pca.shape[1] == 1:
        plt.scatter(X_pca[:,0], np.zeros(len(X_pca)), c=labels)
        plt.xlabel("Principal Component 1")
        plt.yticks([])
        
    else:
        plt.scatter(X_pca[:,0], X_pca[:,1], c=labels)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")

    plt.title(title)
    plt.show()


# =============================
# Load Dataset
# =============================

print("Loading Dataset...")

df = pd.read_csv("Combined_dataset.csv")

print("Dataset Shape:", df.shape)
print(df.head())


# =============================
# Data Cleaning
# =============================

print("\nCleaning Data...")

df.drop_duplicates(inplace=True)

numeric_cols = df.select_dtypes(include=['int64','float64']).columns
categorical_cols = df.select_dtypes(include=['object','string']).columns

df[numeric_cols] = df[numeric_cols].fillna(0)
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

print("Cleaned Dataset Shape:", df.shape)


# =============================
# Convert to Numeric
# =============================

print("\nConverting Columns...")

possible_numeric = [
    'rating',
    'no_of_ratings',
    'discount_price',
    'actual_price'
]

for col in possible_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df.fillna(0, inplace=True)


# =============================
# Feature Engineering
# =============================

print("\nFeature Engineering...")

if 'rating' in df.columns and 'no_of_ratings' in df.columns:
    df['sales_score'] = df['rating'] * df['no_of_ratings']

if 'discount_price' in df.columns and 'actual_price' in df.columns:
    df['discount_percent'] = (
        (df['actual_price'] - df['discount_price']) /
        df['actual_price']
    ) * 100

df.replace([np.inf, -np.inf], 0, inplace=True)


# =============================
# Feature Selection
# =============================

print("\nSelecting Features...")

features = [
    'rating',
    'no_of_ratings',
    'discount_price',
    'actual_price',
    'sales_score',
    'discount_percent'
]

features = [col for col in features if col in df.columns]

X = df[features]

print("Features Used:", features)


# =============================
# Scaling
# =============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# =============================
# PCA
# =============================

pca = PCA(n_components=min(2, X.shape[1]))
X_pca = pca.fit_transform(X_scaled)


# =============================
# DENDROGRAM (Hierarchical)
# =============================

print("\nGenerating Dendrogram...")

linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(12,6))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Products")
plt.ylabel("Distance")
plt.show()


# =============================
# Find Optimal Clusters (KMeans)
# =============================

print("\nFinding Optimal Clusters...")

wcss = []
silhouette_scores = []

range_n = range(2, 10)

for i in range_n:
    kmeans = KMeans(n_clusters=i, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))


plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(range_n, wcss, marker='o')
plt.title("Elbow Method")

plt.subplot(1,2,2)
plt.plot(range_n, silhouette_scores, marker='o')
plt.title("Silhouette Score")

plt.show()


# =============================
# KMeans Clustering
# =============================

print("\nRunning KMeans...")

optimal_clusters = 3

kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

df['KMeans_Cluster'] = kmeans_labels

plot_clusters(X_pca, kmeans_labels, "KMeans Clustering")


# =============================
# Hierarchical Clustering
# =============================

print("\nRunning Hierarchical Clustering...")

hc = AgglomerativeClustering(n_clusters=optimal_clusters)
hc_labels = hc.fit_predict(X_scaled)

df['Hierarchical_Cluster'] = hc_labels

plot_clusters(X_pca, hc_labels, "Hierarchical Clustering")


# =============================
# Gaussian Mixture
# =============================

print("\nRunning Gaussian Mixture...")

gmm = GaussianMixture(n_components=optimal_clusters)
gmm_labels = gmm.fit_predict(X_scaled)

df['Gaussian_Cluster'] = gmm_labels

plot_clusters(X_pca, gmm_labels, "Gaussian Mixture Clustering")


# =============================
# DBSCAN
# =============================

print("\nRunning DBSCAN...")

dbscan = DBSCAN(eps=0.7, min_samples=8)
dbscan_labels = dbscan.fit_predict(X_scaled)

df['DBSCAN_Cluster'] = dbscan_labels

plot_clusters(X_pca, dbscan_labels, "DBSCAN Clustering")


# =============================
# Cluster Summary
# =============================

print("\nCluster Summary...")

cluster_summary = df.groupby('KMeans_Cluster')[features].mean()

print(cluster_summary)


# =============================
# Best Selling Cluster
# =============================

print("\nIdentifying Best Selling Products...")

if 'sales_score' in cluster_summary.columns:
    best_cluster = cluster_summary['sales_score'].idxmax()
else:
    best_cluster = cluster_summary['rating'].idxmax()

print("\nBest Selling Cluster:", best_cluster)

best_products = df[df['KMeans_Cluster'] == best_cluster]

print("\nTop Best Selling Products")
print(best_products.head())


# =============================
# Save Output
# =============================

df.to_csv("clustered_products.csv", index=False)

print("\nSaved clustered_products.csv")


# =============================
# Final Visualization
# =============================

plot_clusters(X_pca, df['KMeans_Cluster'], "Final Product Clusters")

print("\nModel Completed Successfully 🚀")