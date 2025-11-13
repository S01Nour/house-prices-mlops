# src/train.py

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

def train_model(data_dir="data/processed", output_dir="models"):
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

    print("[INFO] Entraînement du modèle LinearRegression...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.pkl")
    joblib.dump(model, model_path)

    print(f"[INFO] Modèle sauvegardé dans : {model_path}")

if __name__ == "__main__":
    train_model()
