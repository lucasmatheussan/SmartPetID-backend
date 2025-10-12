#!/usr/bin/env python3
"""
Script de inicialização do banco de dados para o sistema Pet Biometric API

Este script:
1. Cria as tabelas do banco de dados
2. Verifica a conectividade
3. Opcionalmente limpa dados existentes

Uso:
    python init_database.py [--clean]
    
    --clean: Remove todos os dados existentes antes de recriar as tabelas
"""

import argparse
import sys
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Adicionar o diretório atual ao path para importar módulos locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.database import get_database_url, get_db
from database.models import Base, User, Pet, PetImage, IdentificationLog, LostPetReport
from auth.auth_handler import AuthHandler

def create_database_tables(clean=False):
    """
    Cria as tabelas do banco de dados
    
    Args:
        clean (bool): Se True, remove todas as tabelas antes de recriar
    """
    try:
        # Obter URL do banco de dados
        database_url = get_database_url()
        print(f"Conectando ao banco de dados: {database_url}")
        
        # Criar engine
        engine = create_engine(database_url)
        
        if clean:
            print("🗑️  Removendo tabelas existentes...")
            Base.metadata.drop_all(bind=engine)
            print("✅ Tabelas removidas com sucesso")
        
        # Criar todas as tabelas
        print("🔨 Criando tabelas do banco de dados...")
        Base.metadata.create_all(bind=engine)
        print("✅ Tabelas criadas com sucesso")
        
        # Verificar se as tabelas foram criadas
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
            tables = [row[0] for row in result.fetchall()]
            
        print(f"📋 Tabelas criadas: {', '.join(tables)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao criar tabelas: {e}")
        return False

def create_test_user():
    """
    Cria um usuário de teste para facilitar o desenvolvimento
    """
    try:
        # Obter sessão do banco
        db = next(get_db())
        auth_handler = AuthHandler()
        
        # Verificar se já existe um usuário de teste
        existing_user = db.query(User).filter(User.username == "testuser").first()
        if existing_user:
            print("👤 Usuário de teste já existe")
            return existing_user.id
        
        # Criar usuário de teste
        print("👤 Criando usuário de teste...")
        hashed_password = auth_handler.get_password_hash("testpass123")
        
        test_user = User(
            username="testuser",
            email="test@example.com",
            hashed_password=hashed_password,
            full_name="Test User",
            phone="+55 11 99999-9999"
        )
        
        db.add(test_user)
        db.commit()
        db.refresh(test_user)
        
        print(f"✅ Usuário de teste criado com ID: {test_user.id}")
        print(f"   Username: testuser")
        print(f"   Password: testpass123")
        print(f"   Email: test@example.com")
        
        return test_user.id
        
    except Exception as e:
        print(f"❌ Erro ao criar usuário de teste: {e}")
        return None
    finally:
        db.close()

def verify_database_connection():
    """
    Verifica se a conexão com o banco de dados está funcionando
    """
    try:
        database_url = get_database_url()
        engine = create_engine(database_url)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()
            
        print("✅ Conexão com banco de dados verificada")
        return True
        
    except Exception as e:
        print(f"❌ Erro na conexão com banco de dados: {e}")
        return False

def main():
    """
    Função principal do script
    """
    parser = argparse.ArgumentParser(
        description="Inicializa o banco de dados do Pet Biometric API"
    )
    parser.add_argument(
        "--clean", 
        action="store_true", 
        help="Remove todos os dados existentes antes de recriar as tabelas"
    )
    parser.add_argument(
        "--no-test-user", 
        action="store_true", 
        help="Não cria usuário de teste"
    )
    
    args = parser.parse_args()
    
    print("🚀 Inicializando banco de dados Pet Biometric API")
    print("=" * 50)
    
    # Verificar conexão
    if not verify_database_connection():
        print("❌ Falha na verificação da conexão. Abortando.")
        sys.exit(1)
    
    # Criar tabelas
    if not create_database_tables(clean=args.clean):
        print("❌ Falha na criação das tabelas. Abortando.")
        sys.exit(1)
    
    # Criar usuário de teste (se solicitado)
    if not args.no_test_user:
        test_user_id = create_test_user()
        if test_user_id:
            print(f"\n🎯 Para testar o sistema, use:")
            print(f"   Username: testuser")
            print(f"   Password: testpass123")
    
    print("\n" + "=" * 50)
    print("✅ Inicialização do banco de dados concluída com sucesso!")
    print("\n🔗 Para iniciar o servidor:")
    print("   python -m uvicorn main:app --reload --host 0.0.0.0 --port 8001")
    print("\n📖 Documentação da API:")
    print("   http://localhost:8001/docs")

if __name__ == "__main__":
    main()