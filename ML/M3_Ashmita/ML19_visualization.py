
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parents[2]
data_path = root / "data" / "processed"
figures_path = root / "figures"
figures_path.mkdir(exist_ok=True)

# Load project results
models = pd.read_csv(data_path / "model_comparison_results.csv")
clusters = pd.read_csv(data_path / "final_cluster_summary.csv")
integrated = pd.read_csv(data_path / "final_integrated_state_report.csv")

# 1. Regression model comparison
fig, axes = plt.subplots(1,3,figsize=(15,5))

metrics = ["MAE","RMSE","R2"]
titles = ["Mean Absolute Error","Root Mean Squared Error","R-Squared"]

for ax,metric,title in zip(axes,metrics,titles):
    bars = ax.bar(models["Model"],models[metric])
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x",rotation=45)

    for bar in bars:
        ax.text(
            bar.get_x()+bar.get_width()/2,
            bar.get_height(),
            f"{bar.get_height():.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

plt.suptitle("Regression Model Performance Comparison")
plt.tight_layout()
plt.savefig(figures_path / "ML19_regression_comparison.png",dpi=300)
plt.show()
plt.close()

# 2. Cluster distribution
plt.figure(figsize=(9,6))

bars = plt.bar(
    clusters["Cluster"].astype(str),
    clusters["Total_States"]
)

plt.xlabel("Cluster")
plt.ylabel("Number of States/UTs")
plt.title("Distribution of States/UTs Across Clusters")

for bar in bars:
    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height(),
        int(bar.get_height()),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(figures_path / "ML19_cluster_distribution.png",dpi=300)
plt.show()
plt.close()

# 3. Normal and anomaly distribution
status_counts = integrated["Status"].value_counts().reindex(
    ["Normal","Anomaly"],fill_value=0
)

plt.figure(figsize=(8,6))

bars = plt.bar(
    status_counts.index,
    status_counts.values
)

plt.xlabel("Detection Status")
plt.ylabel("Number of States/UTs")
plt.title("Normal vs Anomalous Crime Patterns")

for bar in bars:
    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height(),
        int(bar.get_height()),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(figures_path / "ML19_anomaly_distribution.png",dpi=300)
plt.show()
plt.close()

print("\nFINAL VISUALIZATION COMPLETED")
print("Regression models compared:",len(models))
print("Clusters analysed:",len(clusters))
print("States/UTs analysed:",len(integrated))
print("\nAll three visualizations saved successfully.")

# output files:
# figures/ML19_regression_comparison.png
# figures/ML19_cluster_distribution.png
# figures/ML19_anomaly_distribution.png

# Regression comparison: Compares MAE, RMSE and R2 across models.
# Cluster distribution: Shows the number of states/UTs in each cluster.
# Anomaly distribution: Compares normal and anomalous patterns.
# Conclusion: These visualizations summarize the major ML results.
# Results are exploratory and do not represent actual safety rankings.
