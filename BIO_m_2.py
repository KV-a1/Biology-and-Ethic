import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# **Load the dataset**
df = pd.read_excel("/Users/nikitha21sree/Python Program/Project/Output/Bio/male inferitility_1_With_Status.xlsx")

# **Drop completely empty columns & rows**
df.dropna(how="all", axis=1, inplace=True)  # Remove empty columns
df.dropna(how="all", axis=0, inplace=True)  # Remove empty rows

# **Select only numeric features for clustering**
exclude_columns = ["ID", "Fertility Score", "Fertility Status"]
features = df.drop(columns=[col for col in exclude_columns if col in df.columns], errors="ignore")

# **Convert all columns to numeric**
features = features.apply(pd.to_numeric, errors='coerce')

# **Fill missing values with column median (Ensures No NaNs)**
features.fillna(features.median(numeric_only=True), inplace=True)

# **Standardize the numerical features**
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# **Check if NaN still exists (Debugging Step)**
if np.isnan(X_scaled).sum() > 0:
    print("⚠️ WARNING: There are still NaN values in the data!")
    exit()

# **Find the optimal number of clusters (Elbow Method)**
inertia = []
k_range = range(1, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# **Plot the Elbow Curve**
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method to Determine Optimal Clusters')
plt.show()

# **Apply K-Means Clustering (Using 3 Clusters)**
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# **Map Cluster Labels to Fertility Status**
cluster_mapping = {0: "Infertile", 1: "Prone to Infertility", 2: "Fertile"}
df["Cluster Label"] = df["Cluster"].map(cluster_mapping)

# **Save the Clustered Dataset**
output_file = "/Users/nikitha21sree/Python Program/Clustered_Male_Fertility_Dataset.xlsx"
df.to_excel(output_file, index=False)
print(f"✅ Clustered dataset saved as '{output_file}'.")

# **PCA for Visualization**
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# **Scatter plot of clusters**
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["PCA1"], y=df["PCA2"], hue=df["Cluster Label"], palette="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA-Based Clustering of Fertility Groups")
plt.legend(title="Fertility Status")
plt.show()
