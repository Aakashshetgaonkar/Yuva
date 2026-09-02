import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Styling
sns.set(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120

# 1. Load dataset
df = sns.load_dataset("titanic")
print("Dataset loaded. Shape:", df.shape)

# 2. Quick overview
print("\n--- Head ---")
print(df.head())
print("\n--- Info ---")
print(df.info())
print("\n--- Describe (numeric) ---")
print(df.describe(include=[np.number]).T)

# 3. Missing values summary
missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.concat([missing, missing_pct], axis=1)
missing_df.columns = ["missing_count", "missing_pct"]
print("\n--- Missing values ---")
print(missing_df[missing_df.missing_count > 0])

# 4. Basic cleaning decisions (non-destructive)
# - Keep original df; create a working copy for plots
data = df.copy()

# Fill or flag missing values for certain analyses
# Age: keep as-is for distribution and use median for group summaries where needed
data["age_filled_median"] = data["age"].fillna(data["age"].median())

# Embarked: fill with mode
data["embarked_filled"] = data["embarked"].fillna(data["embarked"].mode()[0])

# Deck: many missing values; create 'deck_known' flag
data["deck_known"] = data["deck"].notna().astype(int)

# 5. Feature engineering examples
data["family_size"] = data["sibsp"].fillna(0) + data["parch"].fillna(0) + 1
data["is_alone"] = (data["family_size"] == 1).astype(int)

# 6. Univariate distributions
# Age histogram
plt.figure(figsize=(8,5))
sns.histplot(data["age"], bins=30, kde=True, color="steelblue")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("hist_age.png")
plt.close()

# Fare histogram (log scale for skew)
plt.figure(figsize=(8,5))
sns.histplot(data["fare"].dropna(), bins=40, color="darkgreen")
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("hist_fare.png")
plt.close()

# 7. Bivariate / categorical analyses
# Boxplot: Age by passenger class
plt.figure(figsize=(8,6))
sns.boxplot(x="class", y="age", data=data, palette="pastel")
plt.title("Age by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig("box_age_by_class.png")
plt.close()

# Survival rate by sex
surv_by_sex = data.groupby("sex")["survived"].mean().reset_index()
plt.figure(figsize=(6,5))
sns.barplot(x="sex", y="survived", data=surv_by_sex, palette="muted")
plt.ylim(0,1)
plt.title("Survival Rate by Sex")
plt.ylabel("Survival Rate")
plt.xlabel("Sex")
plt.tight_layout()
plt.savefig("bar_survival_by_sex.png")
plt.close()

# Survival rate by class
surv_by_class = data.groupby("class")["survived"].mean().reset_index()
plt.figure(figsize=(7,5))
sns.barplot(x="class", y="survived", data=surv_by_class, order=["First","Second","Third"], palette="Blues")
plt.ylim(0,1)
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.xlabel("Passenger Class")
plt.tight_layout()
plt.savefig("bar_survival_by_class.png")
plt.close()

# Survival heatmap across class and sex
pivot = data.pivot_table(index="class", columns="sex", values="survived", aggfunc="mean")
plt.figure(figsize=(6,4))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Survival Rate by Class and Sex")
plt.tight_layout()
plt.savefig("heatmap_survival_class_sex.png")
plt.close()

# 8. Correlation matrix for numeric features
numeric_cols = ["survived", "age_filled_median", "fare", "sibsp", "parch", "family_size", "is_alone", "deck_known"]
corr = data[numeric_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix (selected numeric features)")
plt.tight_layout()
plt.savefig("heatmap_corr.png")
plt.close()

# 9. Pairplot for selected variables (smaller sample to speed up)
pair_cols = ["survived", "age_filled_median", "fare", "family_size"]
sns.pairplot(data[pair_cols].dropna(), hue="survived", corner=True, plot_kws={"alpha":0.6})
plt.savefig("pairplot.png")
plt.close()

# 10. Additional insights: survival counts by embarkation and age group
# Age bins
age_bins = [0, 12, 18, 35, 60, 100]
age_labels = ["Child","Teen","YoungAdult","Adult","Senior"]
data["age_group"] = pd.cut(data["age_filled_median"], bins=age_bins, labels=age_labels, include_lowest=True)

surv_by_agegroup = data.groupby("age_group")["survived"].mean().reset_index()
plt.figure(figsize=(8,5))
sns.barplot(x="age_group", y="survived", data=surv_by_agegroup, palette="magma")
plt.ylim(0,1)
plt.title("Survival Rate by Age Group")
plt.ylabel("Survival Rate")
plt.xlabel("Age Group")
plt.tight_layout()
plt.savefig("bar_survival_by_agegroup.png")
plt.close()

# 11. Print concise findings to console
print("\n--- Key numeric correlations (abs > 0.1) ---")
for col in corr.columns:
    for idx in corr.index:
        val = corr.loc[idx, col]
        if idx != col and abs(val) > 0.1:
            print(f"{idx} vs {col}: {val:.2f}")

print("\n--- Survival rates summary ---")
print("Overall survival rate:", data["survived"].mean().round(3))
print("Survival by sex:\n", data.groupby("sex")["survived"].mean().round(3))
print("Survival by class:\n", data.groupby("class")["survived"].mean().round(3))
print("Survival by age group:\n", surv_by_agegroup.set_index("age_group").round(3))

print("\nPlots saved as PNG files:")
print("- hist_age.png")
print("- hist_fare.png")
print("- box_age_by_class.png")
print("- bar_survival_by_sex.png")
print("- bar_survival_by_class.png")
print("- heatmap_survival_class_sex.png")
print("- heatmap_corr.png")
print("- pairplot.png")
print("- bar_survival_by_agegroup.png")
