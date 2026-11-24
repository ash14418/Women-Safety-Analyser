
import pandas as pd
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parents[2]
data_path = root / "data" / "processed"

for year in [2023,2024]:
    df = pd.read_csv(data_path / f"state_rates_{year}.csv")

    df["log_cases"] = np.log1p(df["Total_CAW_Cases_" + str(year)])
    df["log_female_population"] = np.log1p(df["female_population_2011"])

    df["crime_rate_per_lakh"] = pd.to_numeric(
        df["crime_rate_per_lakh"],errors="coerce"
    )

    df["high_rate_flag"] = (
        df["crime_rate_per_lakh"] >
        df["crime_rate_per_lakh"].median()
    ).astype(int)

    df.to_csv(data_path / f"state_features_{year}.csv",index=False)

    print("\nyear:",year)
    print(df.head())

print("feature engineering completed")
#uses log transformation: A mathematical transformation used to compress large numerical values
#uses np.log1p()
#log_cases log transformed crime cases
#applies log  transformation to female transformation 
#creates new column log_female_population
#then converts crime rates to numeric format
#uses medain of crime rates as threshold
#high_rate_flag classifies 1 as crime rate is above the median/threshold and 0 as below
#hence 3 new features were created through feature engginering and data observation
#log_cases
#log_female_population
#high_rate_flag
#ouput file: state_feature_2023.csv
#output file: state_feature_2024.csv