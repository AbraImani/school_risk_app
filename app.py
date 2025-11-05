import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from src.database.models import get_session, Eleve, User
from src.models.predicteur import PredicteurDecrochage
import joblib
import os

# Configuration de la page
st.set_page_config(
    page_title="Élite Vigilance - Prévention du Décrochage Scolaire",
    page_icon="🎓",
    layout="wide"
)

MODEL_PATH = "model/modele_decrochage.pkl"
try:
    predicteur = PredicteurDecrochage(MODEL_PATH)
    # st.success(f"Modèle chargé avec succès depuis {MODEL_PATH}")
except Exception as e:
    # Garder l'application démarrable et afficher un message clair dans Streamlit
    predicteur = None
    st.error(f"Impossible de charger le modèle depuis {MODEL_PATH}: {e}")

# Fonction pour l'authentification
def authenticate(username, password):
    session = get_session()
    user = session.query(User).filter_by(username=username).first()
    if user and user.password == password:  # Dans un vrai système, utilisez le hachage des mots de passe
        return user.role
    return None

# Interface de connexion
def login_page():
    st.title("🎓 Élite Vigilance - Connexion")
    
    with st.form("login_form"):
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submit = st.form_submit_button("Se connecter")
        
        if submit:
            role = authenticate(username, password)
            if role:
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.session_state['role'] = role
                st.rerun()
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect")

def display_dashboard():
    st.title("📊 Tableau de Bord")
    
    # Récupération des données
    session = get_session()
    eleves = session.query(Eleve).all()
    df = pd.DataFrame([eleve.to_dict() for eleve in eleves])
    
    # Harmoniser les noms/valeurs des colonnes attendues par les graphiques
    if not df.empty:
        # Créer une colonne numérique 'risque_decrochage' à partir de 'Risque Décrochage' (format texte "xx.x%")
        if 'risque_decrochage' not in df.columns and 'Risque Décrochage' in df.columns:
            try:
                df['risque_decrochage'] = (
                    df['Risque Décrochage']
                    .astype(str)
                    .str.replace('%', '', regex=False)
                    .str.replace(',', '.', regex=False)
                    .astype(float) / 100.0
                )
            except Exception:
                # En cas d'échec de conversion, remplir par NaN
                df['risque_decrochage'] = pd.NA
        
        # Alias pour le niveau scolaire attendu par les graphiques
        if 'niveau_scolaire' not in df.columns and 'Niveau' in df.columns:
            df['niveau_scolaire'] = df['Niveau']
    
    if not df.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_eleves = len(df)
            st.metric("Nombre total d'élèves", total_eleves)
            
        with col2:
            # Protéger si la colonne est absente ou non numérique
            if 'risque_decrochage' in df.columns:
                eleves_risque = len(df[pd.to_numeric(df['risque_decrochage'], errors='coerce') >= 0.5])
            else:
                eleves_risque = 0
            st.metric("Élèves à risque", eleves_risque)
            
        with col3:
            taux_risque = (eleves_risque / total_eleves) * 100 if total_eleves > 0 else 0
            st.metric("Taux de risque", f"{taux_risque:.1f}%")
        
        # Graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='risque_decrochage', 
                             title='Distribution des Risques de Décrochage',
                             labels={'risque_decrochage': 'Probabilité de décrochage'})
            st.plotly_chart(fig)
            
        with col2:
            fig = px.box(df, x='niveau_scolaire', y='risque_decrochage',
                        title='Risque de Décrochage par Niveau Scolaire')
            st.plotly_chart(fig)

def new_prediction_page():
    st.title("🆕 Nouvelle Prédiction")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            id_eleve = st.text_input("ID Élève")
            age = st.number_input("Âge", min_value=12, max_value=25)
            sexe = st.selectbox("Sexe", ["Masculin", "Féminin"])
            niveau = st.selectbox("Niveau Scolaire", ["3ème Humanités", "4ème Humanités"])
            redoublement = st.checkbox("Redoublement année précédente")
            statut_bourse = st.checkbox("Boursier")
            
        with col2:
            moyenne_t1 = st.number_input("Moyenne Trimestre 1", min_value=0.0, max_value=100.0)
            moyenne_t2 = st.number_input("Moyenne Trimestre 2", min_value=0.0, max_value=100.0)
            nb_echecs = st.number_input("Nombre de matières en échec", min_value=0)
            absences_t1 = st.number_input("Absences Trimestre 1", min_value=0)
            absences_t2 = st.number_input("Absences Trimestre 2", min_value=0)
            retards = st.number_input("Nombre de retards", min_value=0)
            sanctions = st.number_input("Nombre de sanctions", min_value=0)
            avis_conseil = st.selectbox("Avis du conseil de classe", 
                                      ["Très Favorable", "Favorable", "Favorable avec mise en garde",
                                       "Passable", "Défavorable", "Très Défavorable"])
        
        submitted = st.form_submit_button("Prédire")
        
        if submitted:
            # Vérifier que le modèle est chargé
            if predicteur is None:
                st.error("Le modèle n'est pas chargé. Impossible de faire une prédiction pour le moment.")
                return
            else:
                # Création du DataFrame pour la prédiction
                data = pd.DataFrame({
                    'age': [age],
                    'redoublement': [redoublement],
                    'statut_bourse': [statut_bourse],
                    'moyenne_t1': [moyenne_t1],
                    'moyenne_t2': [moyenne_t2],
                    'nb_matieres_echec': [nb_echecs],
                    'absences_t1': [absences_t1],
                    'absences_t2': [absences_t2],
                    'retards': [retards],
                    'sanctions': [sanctions],
                    'avis_conseil': [avis_conseil],
                    'sexe': [sexe],
                    'niveau': [niveau]
                })
                
                # Prédiction
                proba = predicteur.predict(data)[0]
                risk_factors = predicteur.get_risk_factors(data)

            # Affichage des résultats
            st.header("Résultats de la prédiction")

            # Déterminer le niveau de risque et les couleurs
            if proba >= 0.7:
                risk_label = "Risque élevé"
                risk_color = "#d9534f"  # rouge
                risk_emoji = "🚨"
            elif proba >= 0.4:
                risk_label = "Risque modéré"
                risk_color = "#f0ad4e"  # orange
                risk_emoji = "⚠️"
            else:
                risk_label = "Faible risque"
                risk_color = "#2e7d32"  # vert
                risk_emoji = "✅"

            c1, c2 = st.columns([1.2, 1.3])

            with c1:
                # Jauge Plotly
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    number={"suffix": "%", "font": {"size": 28}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": risk_color},
                        "steps": [
                            {"range": [0, 40], "color": "#d9f2e4"},
                            {"range": [40, 70], "color": "#fde9cf"},
                            {"range": [70, 100], "color": "#f7d6d6"},
                        ],
                        "threshold": {
                            "line": {"color": risk_color, "width": 4},
                            "thickness": 0.75,
                            "value": proba * 100,
                        },
                    },
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Probabilité de décrochage", "font": {"size": 16}},
                ))
                fig.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                # Badge de niveau de risque
                st.markdown(
                    f"""
                    <div style='margin-top:6px;margin-bottom:12px;'>
                        <span style='display:inline-block;padding:8px 14px;border-radius:999px;background:{risk_color};color:white;font-weight:600;'>
                            {risk_emoji} {risk_label}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.metric("Probabilité", f"{proba:.1%}")
                if risk_factors:
                    st.subheader("Facteurs de risque identifiés")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
            
            # Sauvegarde dans la base de données
            session = get_session()
            new_eleve = Eleve(
                id_eleve=id_eleve,
                annee_scolaire=str(datetime.now().year),
                age=age,
                sexe=sexe,
                niveau_scolaire=niveau,
                redoublement=redoublement,
                statut_bourse=statut_bourse,
                moyenne_t1=moyenne_t1,
                moyenne_t2=moyenne_t2,
                nb_matieres_echec=nb_echecs,
                absences_t1=absences_t1,
                absences_t2=absences_t2,
                retards=retards,
                sanctions=sanctions,
                avis_conseil=avis_conseil,
                risque_decrochage=proba,
                date_prediction=datetime.now().date()
            )
            
            try:
                session.add(new_eleve)
                session.commit()
                st.success("Prédiction enregistrée avec succès!")
            except Exception as e:
                session.rollback()
                st.error(f"Erreur lors de l'enregistrement : {str(e)}")
            finally:
                session.close()

def history_page():
    st.title("📚 Historique des Prédictions")
    
    session = get_session()
    eleves = session.query(Eleve).all()
    
    if eleves:
        df = pd.DataFrame([eleve.to_dict() for eleve in eleves])
        
        # Filtres
        col1, col2, col3 = st.columns(3)
        with col1:
            niveau_filter = st.multiselect(
                "Filtrer par niveau",
                options=df['Niveau'].unique()
            )
        with col2:
            date_filter = st.date_input(
                "Filtrer par date de prédiction",
                value=[datetime.now().date()]
            )
        
        # Application des filtres
        if niveau_filter:
            df = df[df['Niveau'].isin(niveau_filter)]
        
        # Affichage du tableau
        st.dataframe(df)
        
    else:
        st.info("Aucune prédiction enregistrée pour le moment.")

def statistics_page():
    st.title("📈 Statistiques")
    
    session = get_session()
    eleves = session.query(Eleve).all()
    
    if eleves:
        df = pd.DataFrame([eleve.to_dict() for eleve in eleves])
        
        # Préparer colonnes numériques pour les calculs/graphes
        if 'risque_decrochage' not in df.columns and 'Risque Décrochage' in df.columns:
            try:
                df['risque_decrochage'] = (
                    df['Risque Décrochage']
                    .astype(str)
                    .str.replace('%', '', regex=False)
                    .str.replace(',', '.', regex=False)
                    .astype(float) / 100.0
                )
            except Exception:
                df['risque_decrochage'] = pd.NA
        
        # Statistiques générales
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Risque moyen par niveau scolaire")
            risk_by_level = df.groupby('Niveau')['risque_decrochage'].mean()
            fig = px.bar(risk_by_level, 
                        title='Risque moyen par niveau')
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Évolution des moyennes")
            try:
                import statsmodels.api as sm  # noqa: F401
                trend = "ols"
            except Exception:
                trend = None
                st.info("Droite de tendance désactivée (package 'statsmodels' non installé). Pour l'activer: python -m pip install statsmodels")
            fig = px.scatter(
                df,
                x='Moyenne T1',
                y='Moyenne T2',
                color='risque_decrochage',
                title="Évolution des moyennes",
                trendline=trend
            )
            st.plotly_chart(fig)
        
        # Analyses supplémentaires
        st.subheader("Analyse des facteurs de risque")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(df, 
                        x='Avis Conseil', 
                        y='risque_decrochage',
                        title="Risque selon l'avis du conseil")
            st.plotly_chart(fig)
        
        with col2:
            try:
                import statsmodels.api as sm  # noqa: F401
                trend = "ols"
            except Exception:
                trend = None
                st.info("Droite de tendance désactivée (package 'statsmodels' non installé). Pour l'activer: python -m pip install statsmodels")
            fig = px.scatter(
                df,
                x='Absences T2',
                y='risque_decrochage',
                title="Impact des absences sur le risque",
                trendline=trend
            )
            st.plotly_chart(fig)
    
    else:
        st.info("Aucune donnée disponible pour les statistiques.")

# Page principale
def main_page():
    # Menu latéral
    with st.sidebar:
        selected = option_menu(
            "Menu Principal",
            ["Tableau de bord", "Nouvelle Prédiction", "Historique", "Statistiques"],
            icons=['house', 'person-plus', 'clock-history', 'graph-up'],
            menu_icon="cast"
        )
        
        st.sidebar.button("Déconnexion", on_click=lambda: st.session_state.clear())
    
    if selected == "Tableau de bord":
        display_dashboard()
    elif selected == "Nouvelle Prédiction":
        new_prediction_page()
    elif selected == "Historique":
        history_page()
    elif selected == "Statistiques":
        statistics_page()

# Point d'entrée de l'application
if __name__ == "__main__":
    if 'logged_in' not in st.session_state:
        login_page()
    else:
        main_page()