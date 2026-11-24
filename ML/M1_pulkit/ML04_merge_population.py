
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data_path = root / "data" / "processed"

def find_file(pattern):
    files = list(data_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"file not found: {pattern}")
    return files[0]

census = pd.read_csv(find_file("*census_district_population*.csv"))

census["state_key"] = census["state_name"].astype(str).str.upper().str.strip().str.replace("&","AND")

population = census.groupby("state_key",as_index=False)["female_population_2011"].sum()

for year in [2023,2024]:
    df = pd.read_csv(find_file(f"*ncrb_state*{year}*.csv"))

    df["state_key"] = df["State_UT"].astype(str).str.upper().str.strip().str.replace("&","AND")

    merged = pd.merge(df,population,on="state_key",how="left")

    print("\nyear:",year)
    print(merged.head())
    print("matched:",merged["female_population_2011"].notna().sum())
    print("unmatched:",merged.loc[merged["female_population_2011"].isna(),"State_UT"].tolist())

    merged.to_csv(data_path / f"state_merged_{year}.csv",index=False)

print("population merging completed")
#merges the data of 2023 and 2024 according to the females in the census of 2011
#output: state_merged_2023.csv 
#output: state_merged_2024.csv