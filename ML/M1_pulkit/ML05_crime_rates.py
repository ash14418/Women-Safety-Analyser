
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[2]
data_path = root / "data" / "processed"

for year in [2023,2024]:
    df = pd.read_csv(data_path / f"state_merged_{year}.csv")

    case_cols = [
        c for c in df.columns
        if ("case" in c.lower() or "caw" in c.lower())
        and str(year) in c
    ]

    if not case_cols:
        raise ValueError(f"crime case column not found for {year}")

    cases_col = case_cols[0]

    df[cases_col] = pd.to_numeric(df[cases_col],errors="coerce")
    df["female_population_2011"] = pd.to_numeric(
        df["female_population_2011"],errors="coerce"
    )

    df["crime_rate_per_lakh"] = (
        df[cases_col] / df["female_population_2011"]
    ) * 100000

    df["crime_rate_per_lakh"] = df["crime_rate_per_lakh"].replace(
        [float("inf"),float("-inf")],pd.NA
    )

    df.to_csv(data_path / f"state_rates_{year}.csv",index=False)

    print("\nyear:",year)
    print(df[[
        "State_UT",
        cases_col,
        "female_population_2011",
        "crime_rate_per_lakh"
    ]].head())

print("crime rate calculation completed")
#takes the merged population and 2023 and 2024 data to convert it into crime rates
#crime rate= (total cases/female population)x100000 (per lakh)
#output file: state_rates_2023.csv
#output file: state_rates_2024.csv
