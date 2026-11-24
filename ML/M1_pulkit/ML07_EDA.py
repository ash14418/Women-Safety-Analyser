
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

root = Path(__file__).resolve().parents[2]
data_path = root / "data" / "processed"
figures = root / "figures"
figures.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

for year in [2023,2024]:

    df = pd.read_csv(data_path / f"state_features_{year}.csv")

    print(f"\nEDA for {year}")
    print("Shape:",df.shape)
    print("\nFirst 5 rows:\n",df.head())
    print("\nMissing values:\n",df.isnull().sum())
    print("\nStatistical summary:\n",df.describe())

    # 1. Crime Rate Distribution
    rates = pd.to_numeric(df["crime_rate_per_lakh"],errors="coerce").dropna()

    mean_rate = rates.mean()
    median_rate = rates.median()

    fig,ax = plt.subplots(figsize=(9,5))

    ax.hist(rates,bins=10,color="#6C8EBF",edgecolor="white",alpha=0.9)
    ax.axvline(mean_rate,color="#E07A5F",linestyle="--",linewidth=2,label=f"Mean: {mean_rate:.2f}")
    ax.axvline(median_rate,color="#2A9D8F",linestyle="--",linewidth=2,label=f"Median: {median_rate:.2f}")

    ax.set_title(f"Distribution of Crime Rates Across States ({year})",fontsize=14,fontweight="bold",pad=15)
    ax.set_xlabel("Reported cases per lakh women")
    ax.set_ylabel("Number of States / UTs")
    ax.legend()

    fig.tight_layout()
    fig.savefig(figures / f"ML07_rate_distribution_{year}.png",dpi=300)
    plt.show()
    plt.close(fig)

    # 2. Top 10 States by Crime Rate
    top = df.nlargest(10,"crime_rate_per_lakh").sort_values("crime_rate_per_lakh")

    fig,ax = plt.subplots(figsize=(9,6))

    bars = ax.barh(top["State_UT"],top["crime_rate_per_lakh"],color="#4C78A8")
    ax.bar_label(bars,fmt="%.1f",padding=5,fontsize=9)

    ax.set_title(f"Top 10 States / UTs by Crime Rate ({year})",fontsize=14,fontweight="bold",pad=15)
    ax.set_xlabel("Reported cases per lakh women")
    ax.set_ylabel("State / UT")
    ax.set_xlim(0,top["crime_rate_per_lakh"].max()*1.15)

    fig.tight_layout()
    fig.savefig(figures / f"ML07_top_states_{year}.png",dpi=300)
    plt.show()
    plt.close(fig)

    # 3. Correlation Heatmap
    case_col = f"Total_CAW_Cases_{year}"

    columns = [
        case_col,
        "female_population_2011",
        "crime_rate_per_lakh",
        "log_cases",
        "log_female_population",
        "high_rate_flag"
    ]

    available = [col for col in columns if col in df.columns]

    numeric = df[available].apply(pd.to_numeric,errors="coerce")
    corr = numeric.corr()

    label_names = {
        case_col:"Total Cases",
        "female_population_2011":"Female Population",
        "crime_rate_per_lakh":"Crime Rate",
        "log_cases":"Log Cases",
        "log_female_population":"Log Population",
        "high_rate_flag":"High Rate Flag"
    }

    labels = [label_names[col] for col in available]

    fig,ax = plt.subplots(figsize=(9,7))

    image = ax.imshow(corr,cmap="coolwarm",vmin=-1,vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels,rotation=35,ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.iloc[i,j]
            if pd.notna(value):
                ax.text(j,i,f"{value:.2f}",ha="center",va="center",fontsize=9)

    fig.colorbar(image,ax=ax,label="Correlation")
    ax.set_title(f"Correlation Between Selected Features ({year})",fontsize=14,fontweight="bold",pad=15)

    fig.tight_layout()
    fig.savefig(figures / f"ML07_correlation_{year}.png",dpi=300)
    plt.show()
    plt.close(fig)

print("\nEDA completed for 2023 and 2024")


# 1. output image: rate_distribution of both 2023 and 2024

# 2023
# Mean crime rate: 61.48
# Median crime rate: 53.20
# Most states have rates concentrated around 30–85 cases per lakh women.
# A few states have considerably higher rates, reaching approximately 138.
# Mean is higher than median, suggesting a right-skewed distribution.

# 2024
# Mean crime rate: 60.20
# Median crime rate: 49.49
# Most states have rates concentrated around 30–90.
# A few states have much higher rates, reaching approximately 130.
# Mean is again higher than median, indicating right skewness.

# conclusion:
# The average calculated crime rate decreased slightly from 61.48 to 60.20.
# The median also decreased from 53.20 to 49.49.
# Crime rates vary considerably across states, and a few higher values influence the average.


# 2. output image: top_10_states by crime rate (2023)

# Rajasthan has the highest calculated crime rate of 137.7.
# Haryana ranks second with a rate of 132.9.
# Odisha ranks third with a rate of 124.8.
# Madhya Pradesh and Kerala have similar rates of 92.4 and 92.2.
# Maharashtra, Assam, Chandigarh, West Bengal and Uttarakhand complete the top 10.

# conclusion:
# Rajasthan, Haryana and Odisha have the highest calculated crime rates in 2023.
# There is a noticeable gap between the top three states and the remaining states.
# These states can be selected for further investigation.


# 3. output image: correlation_heatmap of 2023 and 2024

# 2023
# Total Cases and Female Population have a strong positive correlation of 0.91.
# Log Cases and Log Population have a very strong positive correlation of 0.96.
# Crime Rate and High Rate Flag have a strong positive correlation of 0.80.
# Female Population and Crime Rate have a weak positive correlation of 0.26.
# Total Cases and High Rate Flag have a weak negative correlation of -0.10.

# 2024
# Total Cases and Female Population have a strong positive correlation of 0.93.
# Log Cases and Log Population have a very strong positive correlation of 0.97.
# Crime Rate and High Rate Flag have a strong positive correlation of 0.79.
# Female Population and Crime Rate have a weak positive correlation of 0.28.
# Total Cases and High Rate Flag have a weak negative correlation of -0.09.

# conclusion:
# Total Cases and Female Population show a strong positive relationship in both years.
# Log Cases and Log Population show the strongest positive relationship.
# Crime Rate and High Rate Flag are strongly correlated because the flag is created using the crime rate.
# The correlation patterns remain largely similar across 2023 and 2024.
# Correlation helps identify relationships between features before ML modelling.


# overall EDA conclusion:
# EDA helped us understand the distribution and variation of reported crime rates.
# It helped identify states with comparatively higher calculated rates.
# Correlation analysis revealed relationships between the selected features.
# The findings help us understand the dataset and prepare for the machine learning stage.
