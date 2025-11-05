from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, model_name=""):
    """
    Évalue les performances d'un modèle
    """
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    # Affichage des résultats
    print(f"\nRésultats pour {model_name}:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    print(f"AUC-ROC: {auc_roc:.3f}")
    
    print("\nRapport de classification détaillé:")
    print(classification_report(y_test, y_pred))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de confusion - {model_name}')
    plt.ylabel('Valeur réelle')
    plt.xlabel('Prédiction')
    
    # Sauvegarder la matrice de confusion
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_")}.png')
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'confusion_matrix': cm
    }

def analyze_feature_importance(model, feature_names, model_name=""):
    """
    Analyse l'importance des caractéristiques pour les modèles qui le supportent
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print("Ce modèle ne supporte pas l'analyse d'importance des caractéristiques")
            return
        
        # Tri des caractéristiques par importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # Affichage textuel des caractéristiques les plus importantes
        print(f"\nImportance des caractéristiques pour {model_name}:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"{row['feature']}: {row['importance']:.3f}")
        
        # Visualisation
        plt.figure(figsize=(12, 6))
        sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
        plt.title(f'Importance des caractéristiques - {model_name}')
        plt.tight_layout()
        
        # Sauvegarder le graphique
        plt.savefig(f'feature_importance_{model_name.replace(" ", "_")}.png')
        plt.close()
        
        return feature_importance
    
    except Exception as e:
        print(f"Erreur lors de l'analyse d'importance des caractéristiques: {str(e)}")
        return None