
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

root = Path(__file__).resolve().parents[2]
data_path = root / "data" / "processed"
figures_path = root / "figures"
figures_path.mkdir(exist_ok=True)

df = pd.read_csv(data_path / "state_clustering_scaled.csv")
df = df[~df["State_UT"].str.startswith("TOTAL",na=False)].copy()
df = df.reset_index(drop=True)

X = df.drop(columns=["State_UT"])

model = IsolationForest(
    contamination=0.1,
    random_state=42
)

df["Anomaly"] = model.fit_predict(X)
df["Anomaly_Score"] = model.decision_function(X)

df["Status"] = df["Anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})

df.to_csv(data_path / "state_anomaly_results.csv",index=False)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(12,8))

normal = df["Anomaly"] == 1
anomaly = df["Anomaly"] == -1

plt.scatter(X_pca[normal,0],X_pca[normal,1],label="Normal",s=80)
plt.scatter(X_pca[anomaly,0],X_pca[anomaly,1],label="Anomaly",s=100,marker="X")

for i,state in enumerate(df["State_UT"]):
    plt.annotate(state,(X_pca[i,0],X_pca[i,1]),fontsize=7)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("State-wise Crime Pattern Anomaly Detection")
plt.legend()
plt.tight_layout()
plt.savefig(figures_path / "ML15_anomaly_detection.png",dpi=300)
plt.show()
plt.close()

print("\nANOMALY DETECTION RESULTS")
print("\nStatus Distribution:")
print(df["Status"].value_counts())

print("\nStates/UTs Identified as Anomalies:")
print(df.loc[df["Status"]=="Anomaly",["State_UT","Anomaly_Score"]].to_string(index=False))

# output files:
# data/processed/state_anomaly_results.csv
# figures/ML15_anomaly_detection.png

# Isolation Forest identifies unusual crime-rate patterns.
# Normal: Pattern is not identified as unusual.
# Anomaly: Pattern is identified as unusual.
# Anomaly does not automatically mean a state is unsafe.
# Conclusion: Results highlight unusual patterns for further investigation.
# Results are exploratory because the dataset contains only 36 states/UTs.
