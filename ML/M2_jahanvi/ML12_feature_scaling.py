
import pandas as pd
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

root = Path(__file__).resolve().parents[2]
data_path = root / "data" / "processed"

df = pd.read_csv(data_path / "state_features_2024.csv")

features = [
    "Rape_Rate_per_lakh",
    "Attempt_Rape_Rate_per_lakh",
    "Molestation_Harassment_Stalking_Rate_per_lakh",
    "Kidnapping_Abduction_Rate_per_lakh",
    "Domestic_Dowry_Rate_per_lakh",
    "Trafficking_Prostitution_Rate_per_lakh",
    "Cyber_Rate_per_lakh",
    "POCSO_Rate_per_lakh",
    "Acid_Attack_Rate_per_lakh"
]

missing = [col for col in features if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

X = df[features].apply(pd.to_numeric,errors="coerce")

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

scaled_df = pd.DataFrame(X_scaled,columns=features)
scaled_df.insert(0,"State_UT",df["State_UT"])

scaled_df.to_csv(data_path / "state_clustering_scaled.csv",index=False)

joblib.dump({
    "imputer":imputer,
    "scaler":scaler,
    "features":features
},data_path / "state_scaler.pkl")

print("\nFEATURE SCALING COMPLETED")
print("Original Shape:",X.shape)
print("Scaled Shape:",X_scaled.shape)
print("\nScaled Data:")
print(scaled_df.head())

print("\nMissing Values:",scaled_df.isnull().sum().sum())
print("\nScaling completed successfully")

# output files:
# state_clustering_scaled.csv: Stores the scaled crime-category features.
# state_scaler.pkl: Stores the imputer, scaler and feature names.

# SimpleImputer: Replaces missing values using the median.
# StandardScaler: Transforms features to have mean 0 and standard deviation 1.
# State_UT: Preserved to identify each state or union territory.

# Conclusion:
# Prepares 2024 state/UT crime-rate features for clustering.
# Scaling ensures that features with larger numerical values
# do not dominate distance-based clustering algorithms.
