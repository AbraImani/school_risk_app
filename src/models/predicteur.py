import joblib
import pandas as pd
import numpy as np
from pathlib import Path

class PredicteurDecrochage:
    def __init__(self, model_path):
        """
        Initialise le prédicteur avec le modèle entraîné
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(str(Path(model_path).parent / 'scaler.pkl'))
        
    def preprocess_data(self, data):
        """
        Prétraite les données pour la prédiction
        """
        # Copie des données
        data_processed = data.copy()
        
        # Renommage des colonnes pour correspondre au modèle entraîné
        mapping = {
            'age': 'Age',
            'redoublement': 'Redoublement_Annee_Precedente',
            'statut_bourse': 'Statut_Bourse',
            'moyenne_t1': 'Moyenne_Generale_T1',
            'moyenne_t2': 'Moyenne_Generale_T2',
            'nb_matieres_echec': 'Nombre_Matieres_Echec_T1',
            'absences_t1': 'Nombre_Absences_Injustifiees_T1',
            'absences_t2': 'Nombre_Absences_Injustifiees_T2',
            'retards': 'Nombre_Retards_T1',
            'sanctions': 'Nombre_Sanctions_Disciplinaires_T1',
            'niveau': 'Niveau_Scolaire_Actuel'
        }
        data_processed.rename(columns=mapping, inplace=True)

        # Encodage du niveau scolaire
        data_processed['Niveau_Scolaire_Actuel_3ème Humanités'] = (data_processed['Niveau_Scolaire_Actuel'] == '3ème Humanités').astype(int)
        data_processed['Niveau_Scolaire_Actuel_4ème Humanités'] = (data_processed['Niveau_Scolaire_Actuel'] == '4ème Humanités').astype(int)
        
        # Encodage du sexe
        data_processed['Sexe_Féminin'] = (data['sexe'] == 'Féminin').astype(int)
        data_processed['Sexe_Masculin'] = (data['sexe'] == 'Masculin').astype(int)
        
        # Création des colonnes pour l'encodage one-hot de l'avis du conseil
        # IMPORTANT: Ordre alphabétique pour correspondre à get_dummies
        avis_columns = [
            'Avis_Conseil_Classe_T1_Défavorable',
            'Avis_Conseil_Classe_T1_Favorable',
            'Avis_Conseil_Classe_T1_Favorable avec mise en garde',
            'Avis_Conseil_Classe_T1_Passable',
            'Avis_Conseil_Classe_T1_Très Défavorable',
            'Avis_Conseil_Classe_T1_Très Favorable'
        ]
        
        # Initialisation des colonnes d'avis à 0
        for col in avis_columns:
            data_processed[col] = 0
            
        # Mapping des avis
        avis_mapping = {
            'Défavorable': 'Avis_Conseil_Classe_T1_Défavorable',
            'Favorable': 'Avis_Conseil_Classe_T1_Favorable',
            'Favorable avec mise en garde': 'Avis_Conseil_Classe_T1_Favorable avec mise en garde',
            'Passable': 'Avis_Conseil_Classe_T1_Passable',
            'Très Défavorable': 'Avis_Conseil_Classe_T1_Très Défavorable',
            'Très Favorable': 'Avis_Conseil_Classe_T1_Très Favorable'
        }
        
        # Application du mapping des avis
        for idx, avis in enumerate(data['avis_conseil']):
            if avis in avis_mapping:
                data_processed.loc[idx, avis_mapping[avis]] = 1
                
        # Création des nouvelles variables
        data_processed['evolution_moyenne'] = data_processed['Moyenne_Generale_T2'] - data_processed['Moyenne_Generale_T1']
        data_processed['evolution_absences'] = data_processed['Nombre_Absences_Injustifiees_T2'] - data_processed['Nombre_Absences_Injustifiees_T1']
        data_processed['total_absences'] = data_processed['Nombre_Absences_Injustifiees_T1'] + data_processed['Nombre_Absences_Injustifiees_T2']
        
        # Suppression des colonnes non utilisées
        columns_to_drop = ['Niveau_Scolaire_Actuel', 'avis_conseil', 'sexe']
        data_processed = data_processed.drop(columns=[col for col in columns_to_drop if col in data_processed.columns])

        # Harmoniser les types des variables booléennes/catégorielles codées en 0/1
        for col in ['Redoublement_Annee_Precedente', 'Statut_Bourse',
                    'Niveau_Scolaire_Actuel_3ème Humanités', 'Niveau_Scolaire_Actuel_4ème Humanités',
                    'Sexe_Féminin', 'Sexe_Masculin'] + avis_columns:
            if col in data_processed.columns:
                data_processed[col] = data_processed[col].astype(int)

        # Standardisation des variables numériques
        numeric_features = ['Age', 'Moyenne_Generale_T1', 'Moyenne_Generale_T2', 
                            'Nombre_Matieres_Echec_T1', 'Nombre_Absences_Injustifiees_T1',
                            'Nombre_Absences_Injustifiees_T2', 'Nombre_Retards_T1',
                            'Nombre_Sanctions_Disciplinaires_T1', 'evolution_moyenne',
                            'evolution_absences', 'total_absences']
        
        data_processed[numeric_features] = self.scaler.transform(data_processed[numeric_features])
        
        # Réorganiser les colonnes dans l'ordre exact attendu par le modèle
        if hasattr(self.model, 'feature_names_in_'):
            expected_columns = list(self.model.feature_names_in_)
        else:
            # Fallback basé sur l'ordre pandas get_dummies dans l'entraînement
            expected_columns = [
                'Age',
                'Redoublement_Annee_Precedente',
                'Moyenne_Generale_T1',
                'Moyenne_Generale_T2',
                'Nombre_Matieres_Echec_T1',
                'Nombre_Absences_Injustifiees_T1',
                'Nombre_Absences_Injustifiees_T2',
                'Nombre_Retards_T1',
                'Nombre_Sanctions_Disciplinaires_T1',
                'Statut_Bourse',
                'evolution_moyenne',
                'evolution_absences',
                'total_absences',
                'Niveau_Scolaire_Actuel_3ème Humanités',
                'Niveau_Scolaire_Actuel_4ème Humanités',
                'Avis_Conseil_Classe_T1_Défavorable',
                'Avis_Conseil_Classe_T1_Favorable',
                'Avis_Conseil_Classe_T1_Favorable avec mise en garde',
                'Avis_Conseil_Classe_T1_Passable',
                'Avis_Conseil_Classe_T1_Très Défavorable',
                'Avis_Conseil_Classe_T1_Très Favorable',
                'Sexe_Féminin',
                'Sexe_Masculin'
            ]

        # S'assurer que toutes les colonnes attendues existent (créer à 0 si manquantes)
        for col in expected_columns:
            if col not in data_processed.columns:
                data_processed[col] = 0

        # Supprimer les colonnes inattendues éventuelles
        data_processed = data_processed[[c for c in expected_columns if c in data_processed.columns]]
        
        data_processed = data_processed[expected_columns]
        
        return data_processed
        
    def predict(self, data):
        """
        Fait une prédiction pour un élève
        """
        # Prétraitement des données
        features = self.preprocess_data(data)
        
        # Prédiction
        proba = self.model.predict_proba(features)
        
        return proba[:, 1]  # Retourne la probabilité de décrochage
        
    def get_risk_factors(self, data):
        """
        Identifie les principaux facteurs de risque
        """
        risk_factors = []
        
        # Évolution des moyennes
        evolution = data['moyenne_t2'].iloc[0] - data['moyenne_t1'].iloc[0]
        if evolution < -5:
            risk_factors.append("Baisse significative des résultats scolaires")
        
        # Moyennes faibles
        if data['moyenne_t2'].iloc[0] < 50:
            risk_factors.append("Moyenne générale insuffisante")
        
        # Matières en échec
        if data['nb_matieres_echec'].iloc[0] > 3:
            risk_factors.append("Nombre élevé de matières en échec")
        
        # Absences
        if data['absences_t2'].iloc[0] > 10:
            risk_factors.append("Taux d'absentéisme élevé")
        if data['absences_t2'].iloc[0] > data['absences_t1'].iloc[0]:
            risk_factors.append("Augmentation des absences")
            
        # Retards
        if data['retards'].iloc[0] > 10:
            risk_factors.append("Nombre important de retards")
            
        # Sanctions
        if data['sanctions'].iloc[0] > 2:
            risk_factors.append("Problèmes de comportement")
            
        # Avis du conseil
        if data['avis_conseil'].iloc[0] in ['Défavorable', 'Très Défavorable']:
            risk_factors.append("Avis défavorable du conseil de classe")
            
        return risk_factors