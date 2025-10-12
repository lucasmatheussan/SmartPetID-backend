import sqlite3
import uuid
from passlib.context import CryptContext
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configurar o contexto de criptografia
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Função para gerar hash da senha
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# Conectar ao banco de dados
conn = sqlite3.connect('pet_biometric.db')
cursor = conn.cursor()

# Dados do novo usuário administrador
new_admin_id = str(uuid.uuid4())
username = "superadmin"
email = "superadmin@focinhoid.com"
password = "admin123"  # Senha clara
hashed_password = get_password_hash(password)  # Senha criptografada
full_name = "Super Administrador"
phone = "+5511999999999"

# Verificar se o usuário já existe
cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
existing_user = cursor.fetchone()

if existing_user:
    print(f"Usuário {username} ou email {email} já existe!")
else:
    # Inserir novo usuário
    cursor.execute(
        "INSERT INTO users (id, username, email, hashed_password, full_name, phone) VALUES (?, ?, ?, ?, ?, ?)",
        (new_admin_id, username, email, hashed_password, full_name, phone)
    )
    conn.commit()
    print(f"Usuário administrador criado com sucesso!")
    print(f"Username: {username}")
    print(f"Password: {password}")
    print(f"ID: {new_admin_id}")

# Fechar conexão
conn.close()