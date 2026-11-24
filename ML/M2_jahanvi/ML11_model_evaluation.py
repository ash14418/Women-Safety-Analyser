
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

root = Path(__file__).resolve().parents[2]
data_path = root / "data" / "processed"
figures_path = root / "figures"
figures_path.mkdir(parents=True,exist_ok=True)

df = pd.read_csv(data_path / "model_comparison_predictions.csv")

actual_candidates = ["Actual","actual","Actual_2024","y_test"]
actual_col = next((col for col in actual_candidates if col in df.columns),None)

if actual_col is None:
    raise ValueError(f"Actual value column not found. Available columns: {list(df.columns)}")

if "Linear Regression" not in df.columns:
    raise ValueError(f"Linear Regression column not found. Available columns: {list(df.columns)}")

actual = df[actual_col]
predicted = df["Linear Regression"]

results = pd.DataFrame({
    "Actual":actual,
    "Predicted":predicted
}).dropna()

actual = results["Actual"]
predicted = results["Predicted"]

mae = mean_absolute_error(actual,predicted)
mse = mean_squared_error(actual,predicted)
rmse = np.sqrt(mse)
r2 = r2_score(actual,predicted)

nonzero = actual != 0
mape = np.mean(np.abs((actual[nonzero]-predicted[nonzero])/actual[nonzero]))*100 if nonzero.any() else np.nan

evaluation = pd.DataFrame({
    "Metric":["MAE","MSE","RMSE","R2 Score","MAPE (%)"],
    "Value":[mae,mse,rmse,r2,mape]
})

evaluation.to_csv(data_path / "final_evaluation_results.csv",index=False)

print("\nFINAL MODEL EVALUATION")
print(evaluation.to_string(index=False))
print("\nEvaluation Records:",len(results))

plt.figure(figsize=(8,6))
plt.scatter(actual,predicted)
minimum = min(actual.min(),predicted.min())
maximum = max(actual.max(),predicted.max())
plt.plot([minimum,maximum],[minimum,maximum],"r--")
plt.xlabel("Actual Crime Rate")
plt.ylabel("Predicted Crime Rate")
plt.title("Actual vs Predicted Crime Rate")
plt.tight_layout()
plt.savefig(figures_path / "ML11_actual_vs_predicted.png")
plt.close()

residuals = actual-predicted

plt.figure(figsize=(8,6))
plt.scatter(predicted,residuals)
plt.axhline(y=0,color="red",linestyle="--")
plt.xlabel("Predicted Crime Rate")
plt.ylabel("Residuals")
plt.title("Residual Plot - Linear Regression")
plt.tight_layout()
plt.savefig(figures_path / "ML11_residuals.png")
plt.close()

print("\nEvaluation completed successfully")
print("Evaluation results and graphs saved")

# output files:
# final_evaluation_results.csv: Stores MAE, MSE, RMSE, R2 and MAPE.
# figures/ML11_actual_vs_predicted.png: Compares actual and predicted crime rates.
# figures/ML11_residuals.png: Displays prediction errors (residuals).

# MAE: Measures average absolute prediction error.
# MSE: Measures average squared prediction error.
# RMSE: Measures prediction error in the original rate scale.
# R2 Score: Measures how much variation is explained by the model.
# MAPE: Measures average percentage prediction error.

# Conclusion:
# Evaluates the selected Linear Regression model using held-out predictions.
# The evaluation helps understand how closely predictions match actual values.
# These results represent preliminary performance on a small test dataset.
