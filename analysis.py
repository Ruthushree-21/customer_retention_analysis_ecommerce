import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

print("All libraries are working")

import pandas as pd

print("Pandas is working")

df = pd.read_excel("AMAZON_DATA_SET.xlsx.xlsx", sheet_name="Cleaned Data")

print("Excel file loaded successfully")
print(df.head())

df.columns = df.columns.str.strip()

print("Missing values per column:")
print(df.isna().sum())

print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates().reset_index(drop=True)

print("Data cleaning completed")

demo_cols = ["Age", "Gender", "Occupation", "Monthly_Income", "Shopping_Frequency"]

print("Demographic variables defined:")
print(demo_cols)

groups = {
    "PB": ["PB1", "PB2", "PB3", "PB4", "PB5", "PB6"],
    "DP": ["DP1", "DP2", "DP3", "DP4", "DP5"],
    "CS": ["CS1", "CS2", "CS3", "CS4", "CS5", "CS6", "CS7"],
    "CR": ["CR1", "CR2", "CR3", "CR4", "CR5", "CR6", "CR7"]
}

for name, cols in groups.items():
    df[f"{name}_Score"] = df[cols].mean(axis=1)

score_cols = ["PB_Score", "DP_Score", "CS_Score", "CR_Score"]

print("Composite scores created successfully")
print(df[score_cols].head())

print("\n--- Cronbach's Alpha ---")

for name, cols in groups.items():
    alpha, ci = pg.cronbach_alpha(data=df[cols])
    print(f"{name}: alpha = {alpha:.3f}, 95% CI = {ci}")
    
print("\n--- Demographic frequencies (%) ---")

for c in demo_cols:
    print(f"\n{c}:")
    print(df[c].value_counts(normalize=True).mul(100).round(1))

print("\n--- Descriptive statistics for composite scores ---")
print(df[score_cols].describe().round(2))

print("\n--- Skewness ---")
print(df[score_cols].skew().round(3))

print("\n--- Kurtosis ---")
print(df[score_cols].kurt().round(3))

corr = df[score_cols].corr(method="pearson")

print("\n--- Correlation Matrix ---")
print(corr.round(3))

plt.figure(figsize=(6, 5))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.title("Correlation between Constructs")
plt.tight_layout()

plt.savefig("correlation_heatmap.png", dpi=200)

plt.show()

# 6. LINEAR REGRESSION (CR ~ PB + DP + CS) + VIF

X = df[["PB_Score", "DP_Score", "CS_Score"]]

X = sm.add_constant(X)

y = df["CR_Score"]

model = sm.OLS(y, X).fit()

print("\n--- Linear Regression Summary ---")
print(model.summary())

# VIF - Multicollinearity check
vif = pd.DataFrame()

vif["Variable"] = X.columns

vif["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print("\n--- Variance Inflation Factor (VIF) ---")
print(vif)

# ---------------------------------------------------------------
# 7. MACHINE LEARNING — HIGH VS LOW RETENTION
# ---------------------------------------------------------------

df["Retention_Class"] = np.where(
    df["CR_Score"] >= df["CR_Score"].median(),
    "High",
    "Low"
)

print("\n--- Retention Class ---")
print(df["Retention_Class"].value_counts())

# Copy data for machine learning
df_ml = df.copy()

# Convert demographic categories into numbers
for col in demo_cols:
    df_ml[col] = LabelEncoder().fit_transform(df_ml[col])

# Select features
features = demo_cols + [
    "PB_Score",
    "DP_Score",
    "CS_Score"
]

X_ml = df_ml[features]

y_ml = LabelEncoder().fit_transform(
    df_ml["Retention_Class"]
)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_ml,
    y_ml,
    test_size=0.25,
    random_state=42,
    stratify=y_ml
)

# ---------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------

print("\n--- Logistic Regression ---")

logit = LogisticRegression(
    max_iter=1000
).fit(X_train, y_train)

pred = logit.predict(X_test)

print(classification_report(y_test, pred))

print(
    "AUC:",
    round(
        roc_auc_score(
            y_test,
            logit.predict_proba(X_test)[:, 1]
        ),
        3
    )
)

# ---------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------

print("\n--- Random Forest ---")

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42
).fit(X_train, y_train)

pred_rf = rf.predict(X_test)

print(classification_report(y_test, pred_rf))

# Feature importance
importances = pd.Series(
    rf.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\n--- Feature Importance ---")
print(importances.round(3))

# ---------------------------------------------------------------
# 8. EXPORT DATA FOR POWER BI
# ---------------------------------------------------------------

export_cols = demo_cols + score_cols + ["Retention_Class"]

df[export_cols].to_csv(
    "amazon_analysis_output.csv",
    index=False
)

print("\nSaved: amazon_analysis_output.csv")
print("Python analysis completed successfully!")