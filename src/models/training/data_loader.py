import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Charge les données et sélectionne les caractéristiques les plus importantes
    """
    df = pd.read_csv(file_path)
    
    # Sélection des caractéristiques les plus pertinentes (10-12 caractéristiques)
    selected_features = [
        'Age',
        'Niveau_Scolaire_Actuel',
        'Redoublement_Annee_Precedente',
        'Moyenne_Generale_T1',
        'Moyenne_Generale_T2',
        'Nombre_Matieres_Echec_T1',
        'Nombre_Absences_Injustifiees_T1',
        'Nombre_Absences_Injustifiees_T2',
        'Nombre_Retards_T1',
        'Nombre_Sanctions_Disciplinaires_T1',
        'Avis_Conseil_Classe_T1',
        'Statut_Bourse',
        'Sexe'
    ]
    
    # Filtrer les colonnes
    df = df[selected_features + ['Statut_Decrochage']]
    
    return df

def preprocess_data(df):
    """
    Prétraite les données pour l'entraînement
    """
    df_processed = df.copy()
    
    # Créer des variables dérivées
    df_processed['evolution_moyenne'] = df_processed['Moyenne_Generale_T2'] - df_processed['Moyenne_Generale_T1']
    df_processed['evolution_absences'] = df_processed['Nombre_Absences_Injustifiees_T2'] - df_processed['Nombre_Absences_Injustifiees_T1']
    df_processed['total_absences'] = df_processed['Nombre_Absences_Injustifiees_T1'] + df_processed['Nombre_Absences_Injustifiees_T2']
    
    # Encodage one-hot pour les variables catégorielles
    df_processed = pd.get_dummies(df_processed, columns=['Niveau_Scolaire_Actuel', 'Avis_Conseil_Classe_T1', 'Sexe'])
    
    # Conversion des variables booléennes
    df_processed['Redoublement_Annee_Precedente'] = df_processed['Redoublement_Annee_Precedente'].astype(int)
    df_processed['Statut_Bourse'] = df_processed['Statut_Bourse'].astype(int)
    
    # Séparation features et target
    X = df_processed.drop('Statut_Decrochage', axis=1)
    y = df_processed['Statut_Decrochage']
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardisation des variables numériques
    scaler = StandardScaler()
    numeric_features = ['Age', 'Moyenne_Generale_T1', 'Moyenne_Generale_T2', 
                       'Nombre_Matieres_Echec_T1', 'Nombre_Absences_Injustifiees_T1',
                       'Nombre_Absences_Injustifiees_T2', 'Nombre_Retards_T1',
                       'Nombre_Sanctions_Disciplinaires_T1', 'evolution_moyenne',
                       'evolution_absences', 'total_absences']
    
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    
    return X_train, X_test, y_train, y_test, scaler