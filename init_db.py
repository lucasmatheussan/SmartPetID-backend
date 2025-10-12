#!/usr/bin/env python3
"""
Script para inicializar o banco de dados
"""

import sys
import os

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.database import init_db, engine
from database.models import Base

def main():
    """Inicializa o banco de dados"""
    try:
        print("Inicializando banco de dados...")
        
        # Criar todas as tabelas
        Base.metadata.create_all(bind=engine)
        
        print("✅ Banco de dados inicializado com sucesso!")
        print("\nTabelas criadas:")
        print("- users")
        print("- pets")
        print("- identification_logs")
        print("- lost_pet_reports")
        
    except Exception as e:
        print(f"❌ Erro ao inicializar banco de dados: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()