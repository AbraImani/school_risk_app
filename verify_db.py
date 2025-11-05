import sqlite3

conn = sqlite3.connect('elite_vigilance.db')
cursor = conn.cursor()

# Lister les tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables dans la base de données:")
for table in tables:
    print(f"  - {table[0]}")

# Lister les utilisateurs
print("\nUtilisateurs créés:")
cursor.execute("SELECT username, role FROM users")
users = cursor.fetchall()
for user in users:
    print(f"  - Username: {user[0]}, Role: {user[1]}")

conn.close()
print("\n✓ Base de données vérifiée avec succès!")
