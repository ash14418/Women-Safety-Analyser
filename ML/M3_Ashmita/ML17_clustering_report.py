
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[2]
data_path = root / "data" / "processed"

df = pd.read_csv(data_path / "cluster_anomaly_analysis.csv")

total_states = len(df)
total_clusters = df["Cluster"].nunique()
total_anomalies = (df["Status"] == "Anomaly").sum()
total_normal = (df["Status"] == "Normal").sum()

cluster_summary = df.groupby("Cluster").agg(
    Total_States=("State_UT","count"),
    Normal_States=("Status",lambda x: (x=="Normal").sum()),
    Anomalies=("Status",lambda x: (x=="Anomaly").sum()),
    Average_Anomaly_Score=("Anomaly_Score","mean")
).reset_index()

cluster_summary["Anomaly_Percentage"] = (
    cluster_summary["Anomalies"] /
    cluster_summary["Total_States"] * 100
).round(2)

anomaly_details = df[df["Status"]=="Anomaly"][
    ["State_UT","Cluster","Anomaly_Score"]
].sort_values("Anomaly_Score")

df.to_csv(data_path / "final_state_clustering_report.csv",index=False)
cluster_summary.to_csv(data_path / "final_cluster_summary.csv",index=False)
anomaly_details.to_csv(data_path / "final_anomaly_details.csv",index=False)

print("\nFINAL CLUSTERING REPORT")
print("\nOverall Summary")
print("-------------------------")
print("Total States/UTs:",total_states)
print("Total Clusters:",total_clusters)
print("Normal States/UTs:",total_normal)
print("Anomalies Detected:",total_anomalies)

print("\nCluster-wise Summary")
print("-------------------------")
print(cluster_summary.to_string(index=False))

print("\nAnomaly Details")
print("-------------------------")
print(anomaly_details.to_string(index=False))

largest_cluster = cluster_summary.loc[
    cluster_summary["Total_States"].idxmax()
]

highest_anomaly_cluster = cluster_summary.loc[
    cluster_summary["Anomaly_Percentage"].idxmax()
]

print("\nKEY FINDINGS")
print("-------------------------")
print(
    "Largest Cluster:",
    int(largest_cluster["Cluster"]),
    "with",
    int(largest_cluster["Total_States"]),
    "states/UTs"
)

print(
    "Highest Anomaly Percentage:",
    int(highest_anomaly_cluster["Cluster"]),
    "with",
    highest_anomaly_cluster["Anomaly_Percentage"],
    "%"
)

print("\nCONCLUSION")

print(
    "K-Means grouped states/UTs into similar crime-rate patterns."
)
print(
    "Isolation Forest identified unusual patterns in 4 states/UTs."
)
print(
    "Cluster 2 has the highest anomaly proportion in this dataset."
)
print(
    "These results are exploratory and do not represent actual safety rankings."
)

# output files:
# data/processed/final_state_clustering_report.csv
# data/processed/final_cluster_summary.csv
# data/processed/final_anomaly_details.csv

# final_state_clustering_report.csv: Complete state-wise clustering and anomaly results.
# final_cluster_summary.csv: Cluster-wise statistics and anomaly percentages.
# final_anomaly_details.csv: List of states/UTs identified as anomalies.
# Conclusion: The report combines clustering and anomaly detection findings.
# Results are exploratory because the analysis uses a limited state-level dataset.
