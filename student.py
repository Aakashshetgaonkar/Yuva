import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
RANDOM_STATE = 42

# 1. LOAD DATA
data = fetch_ucirepo(id=320)
X_raw = data.data.features          # 29 demographic/social/school features
y_raw = data.data.targets           # G1, G2, G3 (period 1, 2 and final grades, 0-20)

df = pd.concat([X_raw, y_raw], axis=1)

print("Shape:", df.shape)
print("\nFirst rows:\n", df.head())
print("\nInfo:")
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum().sum(), "total")
print("\nDescriptive stats (numeric):\n", df.describe())


# 2. PREPROCESSING
#Missing values (this dataset has none, but keep the pattern) ---
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include="object").columns.tolist()

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# IMPORTANT MODELING DECISION ---
df = df.drop(columns=["G1", "G2"])

# Outlier capping (IQR) on remaining numeric features
def cap_outliers_iqr(series, k=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return series.clip(lower, upper)

numeric_feature_cols = [c for c in numeric_cols if c in df.columns and c != "G3"]
for col in numeric_feature_cols:
    df[col] = cap_outliers_iqr(df[col])

# Encode categorical features 
categorical_feature_cols = [c for c in categorical_cols if c in df.columns]
df_encoded = pd.get_dummies(df, columns=categorical_feature_cols, drop_first=True)

# Classification target: Pass/Fail 
# Portuguese grading is 0-20; passing is conventionally >= 10.
df_encoded["Result"] = np.where(df_encoded["G3"] >= 10, "Pass", "Fail")
print("\nPass/Fail distribution:\n", df_encoded["Result"].value_counts())


# 3. EXPLORATORY DATA ANALYSIS (VISUALIZATIONS)

# Target distribution
plt.figure()
sns.histplot(df_encoded["G3"], bins=21, kde=True, color="seagreen")
plt.axvline(10, color="red", linestyle="--", label="Pass threshold (10)")
plt.title("Distribution of Final Grade (G3)")
plt.xlabel("G3 (0-20)")
plt.legend()
plt.tight_layout()
plt.savefig("01_grade_distribution.png", dpi=120)
plt.show()
plt.close()

#Correlation heatmap (top numeric features vs G3)
plt.figure(figsize=(10, 8))
corr_cols = numeric_feature_cols + ["G3"]
corr = df_encoded[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap (numeric features)")
plt.tight_layout()
plt.savefig("02_correlation_heatmap.png", dpi=120)
plt.show()
plt.close()

#Boxplots: G3 by a few key categorical/behavioral features
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col in zip(axes, ["studytime", "failures", "higher"]):
    sns.boxplot(data=df, x=col, y="G3", hue=col, ax=ax, palette="Set2", legend=False)
    ax.set_title(f"Final Grade by {col}")
plt.tight_layout()
plt.savefig("03_boxplots_by_feature.png", dpi=120)
plt.show()
plt.close()

# 3d. Absences vs Grade
plt.figure()
sns.scatterplot(data=df_encoded, x="absences", y="G3", hue="Result",alpha=0.5, palette={"Pass": "seagreen", "Fail": "indianred"})
plt.title("Absences vs Final Grade")
plt.tight_layout()
plt.savefig("04_absences_vs_grade.png", dpi=120)
plt.show()
plt.close()

print("\nEDA plots saved")


# 4. TRAIN/TEST SPLIT + SCALING

feature_cols = [c for c in df_encoded.columns if c not in ["G3", "Result"]]
X = df_encoded[feature_cols]
y_reg = df_encoded["G3"]
y_clf = df_encoded["Result"]

X_train, X_test, yreg_train, yreg_test, yclf_train, yclf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 5. REGRESSION: predict continuous G3

print("\n" )
print("REGRESSION RESULTS (predicting G3, without G1/G2)")

reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
}

reg_results = {}
for name, model in reg_models.items():
    model.fit(X_train_scaled, yreg_train)
    preds = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(yreg_test, preds))
    mae = mean_absolute_error(yreg_test, preds)
    r2 = r2_score(yreg_test, preds)
    reg_results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}

    print(f"\n{name}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE : {mae:.4f}")
    print(f"  R2  : {r2:.4f}")

best_reg = reg_models["Random Forest Regressor"]
preds_best = best_reg.predict(X_test_scaled)

plt.figure()
plt.scatter(yreg_test, preds_best, alpha=0.4, s=15, color="darkorange")
plt.plot([0, 20], [0, 20], "k--", lw=2)
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Random Forest Regressor: Actual vs Predicted")
plt.tight_layout()
plt.savefig("05_regression_actual_vs_pred.png", dpi=120)
plt.show()
plt.close()

# 6. CLASSIFICATION: predict Pass/Fail

print("\n")
print("CLASSIFICATION RESULTS")

clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Random Forest Classifier": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    ),
}

clf_results = {}
for name, model in clf_models.items():
    model.fit(X_train_scaled, yclf_train)
    preds = model.predict(X_test_scaled)

    acc = accuracy_score(yclf_test, preds)
    prec = precision_score(yclf_test, preds, pos_label="Pass", zero_division=0)
    rec = recall_score(yclf_test, preds, pos_label="Pass", zero_division=0)
    f1 = f1_score(yclf_test, preds, pos_label="Pass", zero_division=0)
    clf_results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

    print(f"\n{name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print("  Classification Report:")
    print(classification_report(yclf_test, preds, zero_division=0))

best_clf = clf_models["Random Forest Classifier"]
preds_best_clf = best_clf.predict(X_test_scaled)
cm = confusion_matrix(yclf_test, preds_best_clf, labels=["Fail", "Pass"])

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Random Forest Classifier: Confusion Matrix")
plt.tight_layout()
plt.savefig("06_confusion_matrix.png", dpi=120)
plt.show()
plt.close()

pass_idx = list(best_clf.classes_).index("Pass")
y_prob_pass = best_clf.predict_proba(X_test_scaled)[:, pass_idx]
roc_auc = roc_auc_score((yclf_test == "Pass").astype(int), y_prob_pass)
print(f"\nRandom Forest Classifier ROC-AUC: {roc_auc:.4f}")


# 7. FEATURE IMPORTANCE (both tasks, Random Forest)
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

reg_importance = pd.Series(best_reg.feature_importances_, index=feature_cols).sort_values().tail(15)
reg_importance.plot(kind="barh", ax=axes[0], color="steelblue")
axes[0].set_title("Top 15 Features — Regression (RF)")

clf_importance = pd.Series(best_clf.feature_importances_, index=feature_cols).sort_values().tail(15)
clf_importance.plot(kind="barh", ax=axes[1], color="darkorange")
axes[1].set_title("Top 15 Features — Classification (RF)")

plt.tight_layout()
plt.savefig("07_feature_importance.png", dpi=120)
plt.show()
plt.close()


# SUMMARY
print("\n")
print("SUMMARY")
print("\nRegression:")
print(pd.DataFrame(reg_results).T)
print("\nClassification:")
print(pd.DataFrame(clf_results).T)
print("\nAll plots saved in /mnt/user-data/outputs/")