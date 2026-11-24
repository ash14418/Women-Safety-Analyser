
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[2]
data_path = root / "data" / "processed"

df_2023 = pd.read_csv(data_path / "state_features_2023.csv")
df_2024 = pd.read_csv(data_path / "state_features_2024.csv")

df_2023["year"] = 2023
df_2024["year"] = 2024

df = pd.concat([df_2023,df_2024],ignore_index=True,sort=False)

df["crime_rate_per_lakh"] = pd.to_numeric(
    df["crime_rate_per_lakh"],errors="coerce"
)

df = df.dropna(subset=["crime_rate_per_lakh"])

df = df.sort_values(["year","State_UT"]).reset_index(drop=True)

df.to_csv(data_path / "ml_dataset.csv",index=False)

print("ML dataset created successfully")
print("\nShape:",df.shape)
print("\nColumns:\n",df.columns.tolist())
print("\nFirst 5 rows:\n",df.head())
print("\nLast 5 rows:\n",df.tail())
print("\nYear-wise records:\n",df["year"].value_counts())
print("\nMissing values:\n",df.isnull().sum())

#output file: ml_dataset.csv
# Reads both datasets: Loads the feature-engineered data for 2023 and 2024.
# Adds year column: Identifies which year each record belongs to.
# Combines datasets: Uses pd.concat() to join both datasets row-wise.
# Converts crime rate: Ensures the target column contains numerical values.
# Removes missing target values: Removes rows where crime rate is unavailable.
# Sorts records: Arranges data by year and state.
# Saves dataset: Creates the final ml_dataset.csv file.