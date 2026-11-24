import pandas as pd
import os

df = pd.read_excel("data/raw/DDW_PCA0000_2011_Indiastatedist.xlsx",sheet_name="Sheet1")

print(df.shape)
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print("duplicates:",df.duplicated().sum())

df = df.drop_duplicates()

df.columns = df.columns.str.strip().str.lower()

states = df[df["level"]=="STATE"]
states = states[states["tru"]=="Total"]

state_map = dict(zip(states["state"],states["name"]))

df = df[df["level"]=="DISTRICT"]
df = df[df["tru"]=="Total"]

df["state_name"] = df["state"].map(state_map)

df = df[["state","state_name","district","name","tot_f"]]

df = df.rename(columns={
    "name":"district_name",
    "tot_f":"female_population_2011"
})

df["female_population_2011"] = pd.to_numeric(
    df["female_population_2011"],errors="coerce"
)

df = df.dropna(subset=[
    "state_name","district_name","female_population_2011"
])

df = df[df["female_population_2011"]>=0]

print("cleaned shape:",df.shape)
print(df.head())
print(df.isnull().sum())

os.makedirs("data/processed",exist_ok=True)

df.to_csv("data/processed/census_district_population.csv",index=False)

print("district processing completed")
#preprocesses the census data of 2011 according to female population
#ouptut file: census_district_population.csv in data/processed