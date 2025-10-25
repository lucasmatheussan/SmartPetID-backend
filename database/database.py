from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

# URL do banco de dados
# Para desenvolvimento, usando SQLite
# Para produção, usar MySQL ou PostgreSQL
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///./pet_biometric.db"
)

# Exemplos para produção:
# MySQL: "mysql+pymysql://user:password@host:port/database?charset=utf8mb4"
# PostgreSQL: "postgresql://user:password@host:port/database"

# Configurações de engine compatíveis com MySQL
is_sqlite = DATABASE_URL.startswith("sqlite")
is_mysql = DATABASE_URL.startswith("mysql")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if is_sqlite else {},
    pool_pre_ping=True,               # Evita conexões mortas em MySQL
    pool_recycle=1800                 # Recicla conexões antigas (30 min)
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """Dependência para obter sessão do banco de dados"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Cria todas as tabelas no banco de dados"""
    from .models import Base
    Base.metadata.create_all(bind=engine)

def init_db():
    """Inicializa o banco de dados"""
    create_tables()
    print("Banco de dados inicializado com sucesso!")

if __name__ == "__main__":
    init_db()