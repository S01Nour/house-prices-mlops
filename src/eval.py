# src/eval.py

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(data_dir="data/processed", model_path="models/model.pkl"):
    print("[INFO] Chargement des données nettoyées...")
    df = pd.read_csv(os.path.join(data_dir, "train_clean.csv"))

    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    print("[INFO] Binning pour stratification...")
    y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')

    print("[INFO] Split train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_binned
    )

    print("[INFO] Chargement du modèle...")
    model = joblib.load(model_path)

    import numpy as np

def evaluate(model, X, y, label=""):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)  # <- remplace squared=False
    r2 = r2_score(y, y_pred)
    print(f"Évaluation {label}")
    print(f"  → RMSE : {rmse:.2f}")
    print(f"  → R²   : {r2:.2f}")
    print("-" * 30)


    print("[INFO] Évaluation sur ensemble d'entraînement :")
    evaluate(model, X_train, y_train, "Train")

    print("[INFO] Évaluation sur ensemble de test :")
    evaluate(model, X_test, y_test, "Test")

if __name__ == "__main__":
    evaluate_model()
