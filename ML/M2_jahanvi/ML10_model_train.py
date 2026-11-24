
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression

root = Path(__file__).resolve().parents[2]
data_path = root / "data" / "processed"

df = pd.read_csv(data_path / "ml_dataset.csv")

features = [
    "crime_rate_per_lakh",
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

features = [col for col in features if col in df.columns]

wide = df.pivot(
    index="State_UT",
    columns="year",
    values=features
)

wide.columns = [f"{feature}_{year}" for feature,year in wide.columns]
wide = wide.reset_index()

target = df.pivot(
    index="State_UT",
    columns="year",
    values="crime_rate_per_lakh"
)

wide["target_2024"] = wide["State_UT"].map(target[2024])

selected_features = [f"{col}_2023" for col in features]

wide = wide.dropna(subset=selected_features + ["target_2024"])

X = wide[selected_features]
y = wide["target_2024"]

medians = X.median()
X = X.fillna(medians)

model = LinearRegression()

model.fit(X,y)

joblib.dump({
    "model":model,
    "features":selected_features,
    "medians":medians
},data_path / "best_model.pkl")

print("Best model trained successfully")
print("Selected Model: Linear Regression")
print("Training Records:",len(X))
print("Features:",selected_features)

print("\nModel Coefficients:")
for feature,coefficient in zip(selected_features,model.coef_):
    print(feature,":",round(coefficient,4))

print("\nIntercept:",round(model.intercept_,4))
print("\nModel saved successfully")

# output file:
# best_model.pkl: Stores the trained Linear Regression model,
# selected feature names and median values used for missing data.

# Linear Regression: The best-performing model selected from ML09.
# Model Training: Trains the model using 2023 features to predict 2024 crime rates.
# Coefficients: Show the contribution of each input feature to the prediction.
# Intercept: Represents the model's constant value.
# Joblib: Saves the trained model for future use.

# Conclusion:
# Linear Regression was selected because it achieved the lowest MAE and RMSE
# and the highest R2 score during model comparison.
# The model is retrained using all available state/UT records
# and saved for future crime-rate predictions.