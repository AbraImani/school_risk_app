# Élite Vigilance – Prévention du décrochage scolaire

Application Streamlit pour prédire le risque de décrochage scolaire, stocker les prédictions dans une base SQLite et visualiser des statistiques.

## Sommaire
- Aperçu du projet
- Prérequis
- Installation (Windows / PowerShell)
- Initialiser la base de données
- Entraîner (optionnel comme le meilleur modèle est déjà là)
- Lancer l'application
- Identifiants par défaut
- Structure du projet
- Dépannage (FAQ)

---

## Aperçu
- Frontend: Streamlit (`app.py`)
- Modèle ML: RandomForest (ou meilleur modèle entraîné) + StandardScaler sauvegardés dans `model/`
- Base de données: SQLite (`elite_vigilance.db`) avec SQLAlchemy
- Entraînement: pipeline dans `src/models/training/`

## Prérequis
- Windows (PowerShell)
- Python 3.10+ (3.11/3.12/3.13 selon compatibilité de vos packages)

> Note: Si vous utilisez Python 3.13 et rencontrez des erreurs d'installation de packages (scikit-learn, etc.), utilisez une version de Python compatible (3.10–3.12) via pyenv/conda.

## Installation (Windows / PowerShell)
Ouvrez un terminal PowerShell dans le dossier `school_risk_app` puis exécutez:

```powershell
# 1) Créer et activer un environnement virtuel
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Mettre pip à jour et installer les dépendances
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Initialiser la base de données
Crée les tables `users` et `eleves` et insère 3 utilisateurs par défaut.

```powershell
python src\database\init_db.py
```

Si vous voyez l'erreur "No module named 'src'":
- Le script corrige déjà `PYTHONPATH` en interne. Assurez-vous d'exécuter la commande depuis le dossier `school_risk_app`.


## Entraîner (optionnel)
Par défaut, un modèle pré-entraîné est présent dans `model/`:
- `model/modele_decrochage.pkl`
- `model/scaler.pkl`

Pour ré-entraîner sur vos données:
```powershell
# Placez/ajustez votre CSV dans src/models/training/donnees_eleves_complet.csv
cd src\models\training
python train.py
cd ..\..\..
```
Le meilleur modèle et le scaler sont (re)générés dans `model/`.

> Remarque: l'entraînement utilise plusieurs algorithmes (RandomForest, XGBoost, LightGBM, …). L’installation de `xgboost`/`lightgbm` peut être plus délicate sous Windows. Si besoin, commentez-les dans `model_factory.py` ou n’installez pas ces paquets.


## Lancer l'application
```powershell
streamlit run app.py
```
Le terminal affichera une URL, par ex. http://localhost:8502. Ouvrez-la dans votre navigateur.

## Identifiants par défaut
- admin / admin123 (admin)
- prof / prof123 (enseignant)
- conseiller1 / conseiller123 (conseiller)

## Structure du projet
```
App/
  app.py                      # Application Streamlit
  model/                      # Modèle et scaler sauvegardés
    modele_decrochage.pkl
    scaler.pkl
  elite_vigilance.db          # Base SQLite (créée après init)
  src/
    database/
      models.py               # Modèles SQLAlchemy + session
      init_db.py              # Création DB + utilisateurs par défaut
    models/
      predicteur.py           # Prétraitement + prédiction
      training/
        data_loader.py
        train.py
        model_factory.py
        model_evaluation.py
        check_features.py
        donnees_eleves_complet.csv
```

## Dépannage (FAQ)
- Erreur: `no such table: users`
  - Exécutez `python src\database\init_db.py`.

- Erreur: `ModuleNotFoundError: No module named 'statsmodels'`
  - L’application fonctionne sans, mais pour voir les droites de tendance (trendline="ols"): `python -m pip install statsmodels`.

- Erreur: `The feature names should match those that were passed during fit.`
  - Assurez-vous que le couple modèle+scaler (fichiers `.pkl`) vient du même entraînement. Le prédicteur a été ajusté pour réaligner l’ordre et créer les colonnes manquantes si besoin.

- Le modèle ou dataset ne se charge pas
  - Vérifiez la présence de `model/modele_decrochage.pkl` et `model/scaler.pkl`.
  - Le chemin du modèle est défini dans `app.py` via `MODEL_PATH`.
  - Vérifiez la présence de `src\models\training\donnees_eleves_complet.csv`

- Port occupé / changement de port
  - Lancez avec un autre port: `streamlit run app.py --server.port 8503`

