import pandas as pd

print("--- Loading Project Datasets ---")

# 1. Load the Diabetes Data
try:
    diabetes_df = pd.read_csv(r"C:\Users\ARYAN MAHAKUR\Downloads\archive (1)\diabetes.csv")
    print(f"✅ Diabetes dataset loaded! Size: {diabetes_df.shape}")
except Exception as e:
    print(f"❌ Could not load diabetes.csv: {e}")

# 2. Load the Heart Disease Data
try:
    heart_df = pd.read_csv(r"C:\Users\ARYAN MAHAKUR\Downloads\archive (2)\heart.csv")
    print(f"✅ Heart Disease dataset loaded! Size: {heart_df.shape}")
except Exception as e:
    print(f"❌ Could not load heart.csv: {e}")

# 3. Load the Kidney Disease Data
try:
    kidney_df = pd.read_csv(r"C:\Users\ARYAN MAHAKUR\Downloads\archive (3)\new_model.csv")
    print(f"✅ Kidney Disease dataset loaded! Size: {kidney_df.shape}")
except Exception as e:
    print(f"❌ Could not load kidney.csv: {e}")

print("--------------------------------")