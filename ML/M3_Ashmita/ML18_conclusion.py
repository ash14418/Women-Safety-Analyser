
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[2]
data_path = root / "data" / "processed"
data_path.mkdir(exist_ok=True)

# Load clustering results
clustering = pd.read_csv(
    data_path / "final_state_clustering_report.csv"
)

# Load regression results
model_comparison = pd.read_csv(
    data_path / "model_comparison_results.csv"
)

final_evaluation = pd.read_csv(
    data_path / "final_evaluation_results.csv"
)

# Prepare integrated state-wise report
integrated = clustering[
    ["State_UT","Cluster","Status","Anomaly_Score"]
].copy()

integrated.to_csv(
    data_path / "final_integrated_state_report.csv",
    index=False
)

# Save model comparison and evaluation summaries
model_comparison.to_csv(
    data_path / "integrated_model_comparison.csv",
    index=False
)

final_evaluation.to_csv(
    data_path / "integrated_model_evaluation.csv",
    index=False
)

# Overall clustering summary
cluster_summary = integrated.groupby("Cluster").agg(
    Total_States=("State_UT","count"),
    Anomalies=("Status",lambda x: (x=="Anomaly").sum()),
    Normal=("Status",lambda x: (x=="Normal").sum())
).reset_index()

# Generate project summary
summary_path = data_path / "final_project_summary.txt"

with open(summary_path,"w",encoding="utf-8") as file:
    file.write("WOMEN SAFETY ANALYTICS - FINAL PROJECT SUMMARY\n")
    file.write("="*55 + "\n\n")

    file.write("1. REGRESSION MODEL COMPARISON\n")
    file.write("-"*40 + "\n")
    file.write(model_comparison.to_string(index=False))
    file.write("\n\n")

    file.write("2. FINAL MODEL EVALUATION\n")
    file.write("-"*40 + "\n")
    file.write(final_evaluation.to_string(index=False))
    file.write("\n\n")

    file.write("3. CLUSTERING SUMMARY\n")
    file.write("-"*40 + "\n")
    file.write(cluster_summary.to_string(index=False))
    file.write("\n\n")

    file.write("4. OVERALL FINDINGS\n")
    file.write("-"*40 + "\n")
    file.write(f"Total States/UTs Analysed: {len(integrated)}\n")
    file.write(f"Total Clusters: {integrated['Cluster'].nunique()}\n")
    file.write(
        f"Normal States/UTs: {(integrated['Status']=='Normal').sum()}\n"
    )
    file.write(
        f"Anomalies Detected: {(integrated['Status']=='Anomaly').sum()}\n"
    )

    file.write("\n5. CONCLUSION\n")
    file.write("-"*40 + "\n")
    file.write(
        "Regression models were compared to predict crime rates.\n"
    )
    file.write(
        "K-Means grouped states/UTs based on similar crime-rate patterns.\n"
    )
    file.write(
        "Isolation Forest identified unusual crime-rate patterns.\n"
    )
    file.write(
        "These results are exploratory and do not represent actual "
        "safety rankings or guarantees of individual safety.\n"
    )

print("\nFINAL PROJECT INTEGRATION COMPLETED")

print("\nRegression Model Comparison:")
print(model_comparison.to_string(index=False))

print("\nFinal Model Evaluation:")
print(final_evaluation.to_string(index=False))

print("\nClustering Summary:")
print(cluster_summary.to_string(index=False))

print("\nIntegrated State-wise Report:")
print(integrated.to_string(index=False))

print("\nAll integration files saved successfully.")

# output files:
# data/processed/final_integrated_state_report.csv
# data/processed/integrated_model_comparison.csv
# data/processed/integrated_model_evaluation.csv
# data/processed/final_project_summary.txt

# This file combines regression, model evaluation and clustering results.
# Conclusion: The project integrates crime-rate prediction,
# state-wise clustering and anomaly detection.
# Results are exploratory due to limitations in the available data.
