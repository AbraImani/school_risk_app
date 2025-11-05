from src.models.predicteur import PredicteurDecrochage
import pandas as pd

p = PredicteurDecrochage('model/modele_decrochage.pkl')

df = pd.DataFrame({
    'age':[18],
    'redoublement':[0],
    'statut_bourse':[0],
    'moyenne_t1':[65.0],
    'moyenne_t2':[66.0],
    'nb_matieres_echec':[1],
    'absences_t1':[2],
    'absences_t2':[3],
    'retards':[1],
    'sanctions':[0],
    'avis_conseil':['Favorable'],
    'sexe':['Masculin'],
    'niveau':['3ème Humanités']
})

X = p.preprocess_data(df)
print('Columns (ordered):')
print(X.columns.tolist())
print('\nModel expects:')
try:
    print(p.model.feature_names_in_.tolist())
except Exception as e:
    print('No feature_names_in_ on model:', e)

print('\nAttempting predict:')
print('Proba:', p.predict(df)[0])
