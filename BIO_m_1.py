import pandas as pd

# Load the datasets
male_df = pd.read_excel("/Users/nikitha21sree/Ethics/dataset/male inferitility_1.xlsx")  
factors_df = pd.read_excel("/Users/nikitha21sree/Ethics/dataset/Dataset for factors_male.xlsx")  

# **Ensure all original columns are preserved**
merged_df = male_df.copy()

# Convert numeric columns safely (ignoring ID or categorical fields)
for col in merged_df.columns:
    if merged_df[col].dtype == 'object':  
        merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

# Fill missing values with column medians to preserve original structure
merged_df.fillna(merged_df.median(numeric_only=True), inplace=True)

# **Scoring System Based on Multiple Attributes**
def calculate_fertility_score(row):
    score = 0
    scoring_rules = {
        "Sperm concentration (x10⁶/mL)": (15, 2, -2),
        "Total sperm count (x10⁶)": (39, 2, -2),
        "Ejaculate volume (mL)": (1.5, 2, -2),
        "Sperm vitality (%)": (58, 1, -1),
        "Progressive motility (%)": (32, 2, -2),
        "Normal spermatozoa (%)": (4, 2, -2),
        "Head defects (%)": (30, -1, 1),
        "Midpiece and neck defects (%)": (20, -1, 1),
        "Tail defects (%)": (20, -1, 1),
        "Cytoplasmic droplet (%)": (10, -1, 1),
        "Teratozoospermia index": (1.6, -1, 1),
        "Immotile sperm (%)": (40, -2, 2),
        "High DNA stainability, HDS (%)": (15, -1, 1),
        "DNA fragmentation index, DFI (%)": (30, -2, 2),
    }

    for attribute, (threshold, positive, negative) in scoring_rules.items():
        if attribute in row and not pd.isna(row[attribute]):  
            if row[attribute] >= threshold:
                score += positive
            else:
                score += negative

    return score

# **Apply the Scoring Function**
merged_df["Fertility Score"] = merged_df.apply(calculate_fertility_score, axis=1)

# **Classify Fertility Status Based on Score**
def classify_fertility_by_score(score):
    if score <= -1:
        return "Infertile"
    elif 0<= score <= 2:
        return "Prone to Infertility"
    else:
        return "Fertile"

merged_df["Fertility Status"] = merged_df["Fertility Score"].apply(classify_fertility_by_score)


# Save the final dataset
output_file = "/Users/nikitha21sree/Python Program/Project/Output/Bio/male inferitility_1_With_Status.xlsx"
merged_df.to_excel(output_file, index=False)

print(f"✅ Dataset successfully created with all mapped attributes and saved as '{output_file}'.")