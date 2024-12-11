import pandas as pd

df = pd.read_csv("organized_faq_data.csv")
missing_data = df[df.isnull().any(axis=1)]
if not missing_data.empty:
    print("[WARNING] Missing data detected in rows:")
    print(missing_data)
else:
    print("[INFO] Dataset is clean.")
