import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the processed dataset
df = pd.read_excel("PCOS_Updated_Fertility_Status.xlsx")

# Select relevant columns for clustering (excluding categorical and unnecessary attributes)
irrelevant_columns = ["Patient File No.", "Fertility Status", "Fertility Score"]
features = df.drop(columns=irrelevant_columns, errors='ignore')

# Convert numeric columns to proper format
df_cleaned = features.applymap(lambda x: pd.to_numeric(str(x).replace(".", ""), errors='coerce'))

# Fill missing values with column medians instead of dropping rows
df_cleaned.fillna(df_cleaned.median(), inplace=True)

# Standardize the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cleaned)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
k_range = range(1, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method to Determine Optimal Clusters')
plt.show()

# Apply K-Means clustering with the optimal number of clusters (assuming 3 based on previous work)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Analyze clusters before assigning labels
cluster_means = df.groupby("Cluster").mean(numeric_only=True)
print(cluster_means)

# Map clusters to fertility labels based on analysis
cluster_mapping = {0: "Infertile", 1: "Prone to Infertility", 2: "Fertile"}
df["Cluster Label"] = df["Cluster"].map(cluster_mapping)

# Save the clustered dataset
df.to_excel("PCOS_Clustered_Fertility_Status.xlsx", index=False)

# Use PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# Scatter plot with PCA components
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["PCA1"], y=df["PCA2"], hue=df["Cluster Label"], palette="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA-Based Clustering of Fertility Groups")
plt.legend(title="Fertility Status")
plt.show()
