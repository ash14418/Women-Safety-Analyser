
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

root = Path(__file__).resolve().parents[2]
data_path = root / "data" / "processed"
figure_path = root / "figures"

figure_path.mkdir(exist_ok=True)

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

wide["actual_2024"] = wide["State_UT"].map(target[2024])
wide["previous_year_rate"] = wide["State_UT"].map(target[2023])

selected_features = [f"{col}_2023" for col in features]

wide = wide.dropna(
    subset=selected_features + ["actual_2024","previous_year_rate"]
)

X = wide[selected_features]
y = wide["actual_2024"]

X = X.fillna(X.median())

X_train,X_test,y_train,y_test = train_test_split(
    X,y,
    test_size=0.25,
    random_state=42
)

test_states = wide.loc[X_test.index,"State_UT"]

models = {
    "Linear Regression":LinearRegression(),
    "Random Forest":RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),
    "Gradient Boosting":GradientBoostingRegressor(
        random_state=42
    )
}

results = []

baseline = wide.loc[X_test.index,"previous_year_rate"]

results.append({
    "Model":"Naive Baseline",
    "MAE":mean_absolute_error(y_test,baseline),
    "RMSE":np.sqrt(mean_squared_error(y_test,baseline)),
    "R2":r2_score(y_test,baseline)
})

predictions_data = {
    "State_UT":test_states.values,
    "Actual":y_test.values,
    "Naive Baseline":baseline.values
}

for name,model in models.items():

    model.fit(X_train,y_train)

    predictions = model.predict(X_test)

    results.append({
        "Model":name,
        "MAE":mean_absolute_error(y_test,predictions),
        "RMSE":np.sqrt(mean_squared_error(y_test,predictions)),
        "R2":r2_score(y_test,predictions)
    })

    predictions_data[name] = predictions

results_df = pd.DataFrame(results)

results_df.to_csv(
    data_path / "model_comparison_results.csv",
    index=False
)

predictions_df = pd.DataFrame(predictions_data)

predictions_df.to_csv(
    data_path / "model_comparison_predictions.csv",
    index=False
)

print("MODEL PERFORMANCE COMPARISON")
print(results_df.round(3).to_string(index=False))

best_model = results_df.loc[results_df["MAE"].idxmin(),"Model"]

print("\nBest Model based on MAE:",best_model)

fig,axes = plt.subplots(1,3,figsize=(16,5))

metrics = ["MAE","RMSE","R2"]
titles = ["Mean Absolute Error","Root Mean Squared Error","R2 Score"]

for ax,metric,title in zip(axes,metrics,titles):

    values = results_df[metric]

    bars = ax.bar(
        results_df["Model"],
        values
    )

    ax.set_title(title,fontweight="bold")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x",rotation=35)
    ax.grid(axis="y",alpha=0.3)

    if metric in ["MAE","RMSE"]:
        best_index = values.idxmin()
    else:
        best_index = values.idxmax()

    bars[best_index].set_edgecolor("black")
    bars[best_index].set_linewidth(2)

fig.suptitle(
    "Machine Learning Model Comparison",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    figure_path / "ML09_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nComparison chart saved successfully")
print("Selected model:",best_model)

# output files:
# model_comparison_results.csv: Stores MAE, RMSE and R2 scores of all models.
# model_comparison_predictions.csv: Stores actual and predicted crime rates.
# figures/ML09_model_comparison.png: Displays the performance comparison of all models.

# Naive Baseline: Predicts the 2024 crime rate using the 2023 crime rate.
# Linear Regression: Predicts crime rates by learning linear relationships between features.
# Random Forest: Uses multiple decision trees and combines their predictions.
# Gradient Boosting: Builds decision trees sequentially to reduce prediction errors.

# MAE: Measures the average absolute difference between actual and predicted values.
# RMSE: Measures prediction error and gives more weight to larger errors.
# R2 Score: Measures how well the model explains variations in crime rates.

# Lower MAE and RMSE indicate better performance.
# Higher R2 indicates better predictive fit.

# Best Model: Linear Regression.
# It achieved the lowest MAE (7.058), lowest RMSE (8.298)
# and highest R2 score (0.944) among the four approaches.

# Conclusion:
# Compares four regression approaches to predict 2024 crime rates using 2023 data.
# Linear Regression performed best based on the evaluation metrics.
# The results are preliminary because the dataset contains a limited number
# of state/UT records.