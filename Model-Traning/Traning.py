import joblib
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure the folder exists to save our trained model assets
os.makedirs("models", exist_ok=True)


def train_disease_model(name, df, target_column):
    print(f"\n⚡ Training Model for: {name}...")

    # 1. Clean up column names and strip trailing whitespaces
    df.columns = df.columns.str.strip()

    # 2. Handle Text Targets (specifically for Kidney disease text labels)
    if df[target_column].dtype == "object":
        df[target_column] = (
            df[target_column]
            .str.strip()
            .str.lower()
            .map({"ckd": 1, "notckd": 0, "ckd\t": 1})
        )
        df[target_column] = df[target_column].fillna(1).astype(int)

    # 3. Manual binary mapping for categorical feature text inputs
    for col in df.columns:
        if df[col].dtype == "object" and col != target_column:
            df[col] = (
                df[col]
                .str.strip()
                .str.lower()
                .map(
                    {
                        "yes": 1,
                        "no": 0,
                        "normal": 1,
                        "abnormal": 0,
                        "present": 1,
                        "notpresent": 0,
                    }
                )
            )

    # 4. Separate Features (X) and Target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column].astype(int)

    # 5. Train/Test Split (FIXED: Using 'test_size' parameter)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Preprocessing: Imputing Medians & Scaling Data
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # 7. Model Assembly (Random Forest optimized for medical imbalances)
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)

    # 8. Evaluation Performance Check
    predictions = model.predict(X_test_scaled)
    recall = recall_score(y_test, predictions)
    print(f"🎯 {name} Model Sensitivity/Recall: {recall:.2%}")

    # 9. Pack and Freeze Everything to a .pkl File
    model_assets = {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "X_train_raw": X_train,  # Preserved raw layout specifically for LIME charts
    }

    file_name = f"models/{name.lower().replace(' ', '_')}_assets.pkl"
    joblib.dump(model_assets, file_name)
    print(f"💾 Saved assets to {file_name}")


# --- Control Tower Execution ---
if __name__ == "__main__":
    # 1. Process Diabetes
    try:
        df_dia = pd.read_csv(
            r"C:\Users\ARYAN MAHAKUR\Downloads\archive (1)\diabetes.csv"
        )
        train_disease_model("Diabetes", df_dia, "Outcome")
    except Exception as e:
        print(f"❌ Diabetes processing failed: {e}")

    # 2. Process Heart Disease
    try:
        df_heart = pd.read_csv(
            r"C:\Users\ARYAN MAHAKUR\Downloads\archive (2)\heart.csv"
        )
        train_disease_model("Heart Disease", df_heart, "target")
    except Exception as e:
        print(f"❌ Heart Disease processing failed: {e}")

    # 3. Process Chronic Kidney Disease
    try:
        df_kidney = pd.read_csv(
            r"C:\Users\ARYAN MAHAKUR\Downloads\archive (3)\new_model.csv"
        )
        train_disease_model("Chronic Kidney Disease", df_kidney, "Class")
    except Exception as e:
        print(f"❌ Kidney Disease processing failed: {e}")