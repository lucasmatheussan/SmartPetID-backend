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
# MySQL: "mysql+pymysql://user:password@host:port/database"
# PostgreSQL: "postgresql://user:password@host:port/database"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
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