#!/usr/bin/env python3
"""
Script de inicialização do banco de dados para o MVP Pet Biometric
Cria as tabelas e configura o ambiente inicial
"""

import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base, User, Pet
from database.database import DATABASE_URL
from auth.auth_handler import AuthHandler

def create_database():
    """
    Cria o banco de dados e todas as tabelas
    """
    try:
        # Usar URL do banco configurada
        print(f"Conectando ao banco: {DATABASE_URL}")
        
        # Criar engine
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
        )
        
        # Criar todas as tabelas
        print("Criando tabelas...")
        Base.metadata.create_all(bind=engine)
        print("✅ Tabelas criadas com sucesso!")
        
        return engine
        
    except Exception as e:
        print(f"❌ Erro ao criar banco de dados: {e}")
        return None

def create_test_user(engine):
    """
    Cria um usuário de teste para o MVP
    """
    try:
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = SessionLocal()
        auth_handler = AuthHandler()
        
        # Verificar se já existe usuário de teste
        existing_user = db.query(User).filter(User.email == "test@petbiometric.com").first()
        
        if not existing_user:
            # Criar usuário de teste
            test_user = User(
                username="testuser",
                email="test@petbiometric.com",
                full_name="Usuário de Teste",
                hashed_password=auth_handler.get_password_hash("123456"),
                phone="(11) 99999-9999"
            )
            
            db.add(test_user)
            db.commit()
            db.refresh(test_user)
            
            print("✅ Usuário de teste criado:")
            print(f"   Email: test@petbiometric.com")
            print(f"   Senha: 123456")
            print(f"   ID: {test_user.id}")
        else:
            print("ℹ️  Usuário de teste já existe")
            
        db.close()
        
    except Exception as e:
        print(f"❌ Erro ao criar usuário de teste: {e}")

def check_environment():
    """
    Verifica se as variáveis de ambiente estão configuradas
    """
    print("🔍 Verificando configuração do ambiente...")
    
    required_vars = [
        "DATABASE_URL",
        "SECRET_KEY",
        "ALGORITHM"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Variáveis de ambiente faltando:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n💡 Crie um arquivo .env baseado no .env.example")
        return False
    
    print("✅ Configuração do ambiente OK")
    return True

def main():
    """
    Função principal de inicialização
    """
    print("🚀 Inicializando banco de dados do Pet Biometric MVP")
    print("=" * 50)
    
    # Verificar ambiente
    if not check_environment():
        sys.exit(1)
    
    # Criar banco de dados
    engine = create_database()
    if not engine:
        sys.exit(1)
    
    # Criar usuário de teste
    create_test_user(engine)
    
    print("\n🎉 Inicialização concluída com sucesso!")
    print("\n📋 Próximos passos:")
    print("1. Instalar dependências: pip install -r requirements.txt")
    print("2. Iniciar servidor: uvicorn main:app --reload")
    print("3. Acessar documentação: http://localhost:8000/docs")
    print("4. Fazer login com: test@petbiometric.com / 123456")

if __name__ == "__main__":
    main()