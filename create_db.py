#!/usr/bin/env python3

import sqlite3
import os

# Remove existing database
if os.path.exists('pet_biometric.db'):
    os.remove('pet_biometric.db')
    print("Banco de dados existente removido.")

# Create new database and tables
conn = sqlite3.connect('pet_biometric.db')
cursor = conn.cursor()

# Create users table
cursor.execute('''
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    phone VARCHAR(20),
    address TEXT,
    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1
)
''')

# Create pets table
cursor.execute('''
CREATE TABLE pets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL,
    species VARCHAR(50) NOT NULL,
    breed VARCHAR(100),
    age INTEGER,
    description TEXT,
    owner_contact VARCHAR(100),
    biometric_embedding BLOB,
    biometric_confidence REAL,
    owner_id INTEGER,
    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_lost BOOLEAN DEFAULT 0,
    lost_date TIMESTAMP,
    found_date TIMESTAMP,
    last_seen_latitude REAL,
    last_seen_longitude REAL,
    last_seen_address TEXT,
    FOREIGN KEY (owner_id) REFERENCES users (id)
)
''')

# Create pet_images table
cursor.execute('''
CREATE TABLE pet_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pet_id INTEGER NOT NULL,
    image_path VARCHAR(255) NOT NULL,
    biometric_embedding BLOB,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_primary BOOLEAN DEFAULT 0,
    FOREIGN KEY (pet_id) REFERENCES pets (id)
)
''')

conn.commit()
conn.close()

print("Banco de dados criado com sucesso!")
print("Tabelas criadas: users, pets, pet_images")