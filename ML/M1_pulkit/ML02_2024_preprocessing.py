import pandas as pd
import os

df = pd.read_excel("data/raw/Women_Safety_NCRB_2024_Data.xlsx",sheet_name="State_Features_2024")

print(df.shape)
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print("duplicates:",df.duplicated().sum())

df = df.drop_duplicates()
df.columns = df.columns.str.strip()

df = df.dropna(subset=["State_UT"])

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

numeric_cols = df.select_dtypes(include="number").columns
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric,errors="coerce")

df = df.dropna(subset=["Total_CAW_Cases_2024"])
df = df[df["Total_CAW_Cases_2024"]>=0]

os.makedirs("data/processed",exist_ok=True)

df.to_csv("data/processed/ncrb_state_2024_clean.csv",index=False)

city = pd.read_excel("data/raw/Women_Safety_NCRB_2024_Data.xlsx",sheet_name="City_Features_2024")

city = city.drop_duplicates()
city = city.dropna(subset=["City"])
city.to_csv("data/processed/ncrb_city_2024_clean.csv",index=False)

print("2024 preprocessing completed")
#preprocess the 2024 data of women crime ncrb 
#output file:ncrb_state_2024_clean.csv
#output file: ncrb_City_2024_clean.csv