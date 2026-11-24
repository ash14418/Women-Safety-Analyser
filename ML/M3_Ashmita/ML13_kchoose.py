
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans

root = Path(__file__).resolve().parents[2]
data_path = root / "data" / "processed"
figures_path = root / "figures"
figures_path.mkdir(parents=True,exist_ok=True)

df = pd.read_csv(data_path / "state_clustering_scaled.csv")

X = df.drop(columns=["State_UT"])

inertias = []

for k in range(1,11):
    model = KMeans(n_clusters=k,random_state=42,n_init=10)
    model.fit(X)
    inertias.append(model.inertia_)

results = pd.DataFrame({
    "K":range(1,11),
    "Inertia":inertias
})

results.to_csv(data_path / "kmeans_elbow_results.csv",index=False)

plt.figure(figsize=(8,6))
plt.plot(results["K"],results["Inertia"],marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.xticks(range(1,11))
plt.grid(True)
plt.tight_layout()
plt.savefig(figures_path / "ML13_elbow_method.png")
plt.show()

print("\nELBOW METHOD RESULTS")
print(results.to_string(index=False))
print("\nResults saved successfully")

# output files:
# kmeans_elbow_results.csv: Stores K values and their inertia scores.
# figures/ML13_elbow_method.png: Displays the Elbow Method graph.

# K: Represents the number of clusters.
# Inertia: Measures the sum of squared distances between data points
# and their assigned cluster centers.
# Elbow Method: Helps identify a suitable number of clusters
# by observing where the decrease in inertia starts slowing down.

# Conclusion:
# Uses the Elbow Method to identify a suitable K value
# for grouping states/UTs based on their crime-rate patterns.
