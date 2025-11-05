import os
import joblib
from data_loader import load_data, preprocess_data
from model_factory import get_models, train_model
from model_evaluation import evaluate_model, analyze_feature_importance

def main():
    # Charger les données
    data_path = "donnees_eleves_complet.csv"
    print("Chargement des données...")
    df = load_data(data_path)
    
    # Prétraitement des données
    print("\nPrétraitement des données...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Obtenir tous les modèles à tester
    print("\nInitialisation des modèles...")
    models = get_models()
    
    # Stocker les résultats et les modèles entraînés
    results = {}
    trained_models = {}
    
    # Entraîner et évaluer chaque modèle
    for name, model in models.items():
        print(f"\nEntraînement du modèle {name}...")
        trained_model = train_model(model, X_train, y_train)
        trained_models[name] = trained_model
        
        print(f"\nÉvaluation du modèle {name}...")
        metrics = evaluate_model(trained_model, X_test, y_test, name)
        results[name] = metrics
        
        # Analyser l'importance des caractéristiques
        print(f"\nAnalyse de l'importance des caractéristiques pour {name}...")
        feature_importance = analyze_feature_importance(trained_model, X_train.columns, name)
    
    # Sélectionner le meilleur modèle basé sur l'AUC-ROC
    best_model_name = max(results, key=lambda k: results[k]['auc_roc'])
    best_model = trained_models[best_model_name]
    
    print(f"\nMeilleur modèle : {best_model_name}")
    print(f"Score AUC-ROC : {results[best_model_name]['auc_roc']:.3f}")
    
    # Sauvegarder le meilleur modèle et le scaler
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    print(f"Base directory: {base_dir}")
    model_dir = os.path.join(base_dir, "model")
    print(f"Model directory: {model_dir}")
    
    # S'assurer que le répertoire model existe
    os.makedirs(model_dir, exist_ok=True)
    
    # Sauvegarder le modèle
    model_path = os.path.join(model_dir, "modele_decrochage.pkl")
    print(f"Model path: {model_path}")
    joblib.dump(best_model, model_path)
    print(f"Meilleur modèle sauvegardé dans {model_path}")
    
    # Sauvegarder le scaler
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    print(f"Scaler path: {scaler_path}")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler sauvegardé dans {scaler_path}")

if __name__ == "__main__":
    main()