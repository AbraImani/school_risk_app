import pandas as pd
import os
from data_loader import load_data, preprocess_data

# Chemin absolu vers le fichier CSV
csv_path = "donnees_eleves_complet.csv"
# Charger et prétraiter les données d'entraînement
data = load_data(csv_path)
X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

# Afficher toutes les colonnes
print("Colonnes du modèle entraîné :")
print("\n".join(sorted(X_train.columns)))