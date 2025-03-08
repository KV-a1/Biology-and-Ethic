import pandas as pd
import numpy as np

# Load datasets
df_factors = pd.read_excel("/Users/nikitha21sree/Ethics/dataset/Dataset for factors.xlsx")
df_pcos = pd.read_excel("/Users/nikitha21sree/Ethics/dataset/Pcos_Dataset.xlsx")

# Convert factor ranges to numeric values
factor_ranges = df_factors.set_index("Attributes")[["Healthy Range", "Low Range", "High Range"]].apply(pd.to_numeric, errors="coerce").dropna()

# Identify valid attributes by exact matching with factor dataset
valid_attributes = [col for col in df_pcos.columns if col in factor_ranges.index]

# Define weighted attributes using only valid ones
weights = {attr: 2 for attr in valid_attributes}  # Assigning equal weight initially

# Compute fertility score
def calculate_fertility_score(row):
    score = 0
    for attribute, weight in weights.items():
        if attribute in factor_ranges.index and attribute in row:
            try:
                value = float(row[attribute])
                low_range = factor_ranges.loc[attribute, "Low Range"]
                high_range = factor_ranges.loc[attribute, "High Range"]
                
                if low_range <= value <= high_range:
                    score += weight  # Positive score for normal range
                else:
                    score -= weight  # Negative score if out of range
            except ValueError:
                continue
    
    # Apply additional negative score if PCOS (Y/N) is 1
    if "PCOS (Y/N)" in row and row["PCOS (Y/N)"] == 1:
        score -= 1  # Deduct additional points for PCOS positive cases
    
    # Apply scoring for pregnancy and abortions
    if "Pregnant(Y/N)" in row and row["Pregnant(Y/N)"] == 1:
        score -= 1  # Increase score for those who have been pregnant
    if "No. of abortions" in row and row["No. of abortions"] == 1:
        score -= 2  # Decrease score for those with a history of abortion
    
    # Apply positive score for regular exercise
    if "Reg.Exercise(Y/N)" in row and row["Reg.Exercise(Y/N)"] == 1:
        score += 3  # Increase score for those who exercise regularly
    
    
    
    return score

# Apply scoring to dataset
df_pcos["Fertility Score"] = df_pcos.apply(calculate_fertility_score, axis=1)

# Define classification logic
def classify_fertility(score):
    if score >= 2:
        return "Fertile"
    elif score <= -2:
        return "Infertile"
    else:
        return "Prone to Infertility"

# Assign fertility labels
df_pcos["Fertility Status"] = df_pcos["Fertility Score"].apply(classify_fertility)

# Save the updated dataset
df_pcos.to_excel("PCOS_Updated_Fertility_Status.xlsx", index=False)

