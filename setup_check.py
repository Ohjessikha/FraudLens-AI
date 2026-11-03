
from pathlib import Path

import pandas as pd
import numpy as np
import sklearn
import joblib
import streamlit
import matplotlib
import requests

print("=== Fraud Detection Project Setup Check ===")

print("Python environment: Working")
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("Joblib version:", joblib.__version__)
print("Streamlit version:", streamlit.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Requests version:", requests.__version__)

dataset_path = Path("data") / "creditcard.csv"

if dataset_path.exists():
    print("\nDataset file found!")

    df = pd.read_csv(dataset_path, nrows=5)

    print("Columns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.to_string(index=False))

    if "Class" in df.columns:
        print("\nDataset label column found: Class")
        print("Dataset setup check: PASSED")
    else:
        print("\nERROR: Expected 'Class' column was not found.")
else:
    print("\nDataset not found at:", dataset_path)
    print("Download and place creditcard.csv inside the data folder.")
