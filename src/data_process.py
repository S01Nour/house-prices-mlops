# src/data_process.py

import pandas as pd
import numpy as np
import os

def prepare_data(input_path="data/train.csv", output_dir="data/processed"):
    print("[INFO] Chargement des données...")
    df = pd.read_csv(input_path)

    print("[INFO] Nettoyage : colonnes numériques uniquement + suppression des NaN...")
    df_num = df.select_dtypes(include=[np.number])
    df_clean = df_num.dropna()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "train_clean.csv")
    df_clean.to_csv(output_path, index=False)
    print(f"[INFO] Données nettoyées sauvegardées dans : {output_path}")

if __name__ == "__main__":
    prepare_data()
