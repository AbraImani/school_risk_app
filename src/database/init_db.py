import sys
import os

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.database.models import init_db, User, get_session

def create_default_users():
    """Crée les utilisateurs par défaut"""
    session = get_session()
    
    # Vérifier si les utilisateurs existent déjà
    if session.query(User).first() is None:
        # Créer les utilisateurs par défaut
        users = [
            User(username='admin', password='admin123', role='admin'),
            User(username='prof1', password='prof123', role='enseignant'),
            User(username='conseiller1', password='conseiller123', role='conseiller')
        ]
        
        # Ajouter les utilisateurs à la base de données
        for user in users:
            session.add(user)
        
        session.commit()
        print("Utilisateurs par défaut créés avec succès!")
    else:
        print("Les utilisateurs existent déjà.")
    
    session.close()

if __name__ == "__main__":
    print("Initialisation de la base de données...")
    init_db()
    print("Base de données créée avec succès!")
    
    print("\nCréation des utilisateurs par défaut...")
    create_default_users()