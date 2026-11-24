
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parents[2]
data_path = root / "data" / "processed"
figures_path = root / "figures"
figures_path.mkdir(exist_ok=True)

clusters = pd.read_csv(data_path / "state_kmeans_clusters.csv")
anomalies = pd.read_csv(data_path / "state_anomaly_results.csv")

anomaly_data = anomalies[["State_UT","Anomaly","Anomaly_Score","Status"]]

df = clusters[["State_UT","Cluster"]].merge(
    anomaly_data,
    on="State_UT",
    how="inner"
)

df.to_csv(data_path / "cluster_anomaly_analysis.csv",index=False)

print("\nCLUSTER AND ANOMALY ANALYSIS")
print("\nCombined Results:")
print(df.sort_values(["Cluster","State_UT"]).to_string(index=False))

print("\nAnomaly Distribution Across Clusters:")
print(pd.crosstab(df["Cluster"],df["Status"]))

cluster_status = pd.crosstab(df["Cluster"],df["Status"])
cluster_status = cluster_status.reindex(columns=["Normal","Anomaly"],fill_value=0)

ax = cluster_status.plot(
    kind="bar",
    figsize=(10,6),
    width=0.7
)

plt.xlabel("K-Means Cluster")
plt.ylabel("Number of States/UTs")
plt.title("Anomaly Distribution Across K-Means Clusters")
plt.xticks(rotation=0)
plt.legend(title="Status")
plt.tight_layout()
plt.savefig(figures_path / "ML16_cluster_anomaly_analysis.png",dpi=300)
plt.show()
plt.close()

print("\nTotal States/UTs:",len(df))
print("Total Anomalies:",(df["Status"]=="Anomaly").sum())

# output files:
# data/processed/cluster_anomaly_analysis.csv
# figures/ML16_cluster_anomaly_analysis.png

# K-Means groups states/UTs with similar crime-rate patterns.
# Isolation Forest identifies unusual crime-rate patterns.
# This analysis combines both outputs to examine anomalies within each cluster.
# Conclusion: The combined results help identify unusual patterns within groups.
# Clusters are not safety rankings, and anomalies do not automatically mean unsafe.
