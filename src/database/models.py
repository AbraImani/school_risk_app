from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)  # Dans un vrai système, utiliser le hachage
    role = Column(String, nullable=False)  # 'admin', 'enseignant', 'conseiller'
    
class Eleve(Base):
    __tablename__ = 'eleves'
    
    id = Column(Integer, primary_key=True)
    id_eleve = Column(String, unique=True, nullable=False)
    annee_scolaire = Column(String, nullable=False)
    date_prediction = Column(Date, default=datetime.now().date())
    
    # Informations personnelles
    age = Column(Integer, nullable=False)
    sexe = Column(String, nullable=False)
    niveau_scolaire = Column(String, nullable=False)
    redoublement = Column(Boolean, default=False)
    statut_bourse = Column(Boolean, default=False)
    
    # Résultats scolaires
    moyenne_t1 = Column(Float, nullable=False)
    moyenne_t2 = Column(Float, nullable=False)
    nb_matieres_echec = Column(Integer, nullable=False)
    
    # Comportement
    absences_t1 = Column(Integer, nullable=False)
    absences_t2 = Column(Integer, nullable=False)
    retards = Column(Integer, nullable=False)
    sanctions = Column(Integer, nullable=False)
    
    # Avis et prédictions
    avis_conseil = Column(String, nullable=False)
    risque_decrochage = Column(Float, nullable=False)
    
    def to_dict(self):
        """Convertit l'objet en dictionnaire pour l'affichage"""
        return {
            'ID Élève': self.id_eleve,
            'Année Scolaire': self.annee_scolaire,
            'Âge': self.age,
            'Sexe': self.sexe,
            'Niveau': self.niveau_scolaire,
            'Redoublement': 'Oui' if self.redoublement else 'Non',
            'Boursier': 'Oui' if self.statut_bourse else 'Non',
            'Moyenne T1': f"{self.moyenne_t1:.1f}",
            'Moyenne T2': f"{self.moyenne_t2:.1f}",
            'Matières en échec': self.nb_matieres_echec,
            'Absences T1': self.absences_t1,
            'Absences T2': self.absences_t2,
            'Retards': self.retards,
            'Sanctions': self.sanctions,
            'Avis Conseil': self.avis_conseil,
            'Risque Décrochage': f"{self.risque_decrochage:.1%}",
            'Date Prédiction': self.date_prediction.strftime('%d/%m/%Y')
        }

# Création de la base de données
def init_db():
    engine = create_engine('sqlite:///elite_vigilance.db')
    Base.metadata.create_all(engine)
    return engine

def get_session():
    engine = create_engine('sqlite:///elite_vigilance.db')
    Session = sessionmaker(bind=engine)
    return Session()