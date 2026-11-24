
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

root = Path(__file__).resolve().parents[2]
data_path = root / "data" / "processed"
figures_path = root / "figures"
figures_path.mkdir(parents=True,exist_ok=True)

df = pd.read_csv(data_path / "state_clustering_scaled.csv")

df = df[~df["State_UT"].str.startswith("TOTAL",na=False)].copy()
df = df.reset_index(drop=True)

X = df.drop(columns=["State_UT"])

k = 4

model = KMeans(n_clusters=k,random_state=42,n_init=10)
clusters = model.fit_predict(X)

df["Cluster"] = clusters

df.to_csv(data_path / "state_kmeans_clusters.csv",index=False)

joblib.dump(model,data_path / "kmeans_model.pkl")

print("\nK-MEANS CLUSTERING RESULTS")
print("Selected K:",k)
print("Total States/UTs:",len(df))

print("\nCluster Distribution:")
print(df["Cluster"].value_counts().sort_index())

print("\nState-wise Cluster Assignment:")
print(df[["State_UT","Cluster"]].sort_values("Cluster").to_string(index=False))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

centroids_pca = pca.transform(model.cluster_centers_)

plt.figure(figsize=(12,8))

for cluster in range(k):
    mask = clusters == cluster
    plt.scatter(
        X_pca[mask,0],
        X_pca[mask,1],
        label=f"Cluster {cluster}",
        s=70
    )

plt.scatter(
    centroids_pca[:,0],
    centroids_pca[:,1],
    marker="X",
    s=200,
    color="black",
    label="Centroids"
)

for i,state in enumerate(df["State_UT"]):
    plt.annotate(
        state,
        (X_pca[i,0],X_pca[i,1]),
        fontsize=7,
        xytext=(4,4),
        textcoords="offset points"
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering of States and UTs")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(figures_path / "ML14_kmeans_clusters.png",dpi=300)
plt.show()
plt.close()


cluster_counts = df["Cluster"].value_counts().sort_index()

plt.figure(figsize=(8,6))
bars = plt.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

plt.xlabel("Cluster")
plt.ylabel("Number of States/UTs")
plt.title("Number of States/UTs in Each Cluster")

for bar in bars:
    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height(),
        int(bar.get_height()),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(figures_path / "ML14_cluster_distribution.png",dpi=300)
plt.show()
plt.close()

print("\nClustering completed successfully")
print("Cluster visualization saved successfully")

# output files:
# state_kmeans_clusters.csv: Stores state-wise cluster assignments.
# kmeans_model.pkl: Stores the trained K-Means model.
# figures/ML14_kmeans_clusters.png: Displays the four clusters using PCA.

# K-Means: Groups states/UTs based on similar crime-rate patterns.
# PCA: Reduces nine features into two dimensions for visualization.
# Centroids: Represent the center of each cluster.
# fit_predict(): Trains the model and assigns cluster labels.

# Conclusion:
# Groups individual states/UTs into four clusters based on scaled
# crime-category rates.
# PCA provides a two-dimensional visualization of the clusters.
# Cluster numbers represent groups, not safety rankings.
