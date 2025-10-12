from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, LargeBinary, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    phone = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relacionamento com pets
    pets = relationship("Pet", back_populates="owner")

class Pet(Base):
    __tablename__ = "pets"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    species = Column(String(20), nullable=False)  # "dog" ou "cat"
    breed = Column(String(100), nullable=True)
    age = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)
    owner_contact = Column(String(100), nullable=False)
    
    # Dados biométricos
    biometric_embedding = Column(LargeBinary, nullable=False)  # Embedding vetorial serializado
    biometric_confidence = Column(Float, nullable=True)
    
    # Relacionamento com usuário
    owner_id = Column(String, ForeignKey("users.id"), nullable=False)
    owner = relationship("User", back_populates="pets")
    
    # Relacionamento com imagens
    images = relationship("PetImage", back_populates="pet", cascade="all, delete-orphan")
    
    # Dados de registro
    registration_date = Column(DateTime, default=datetime.utcnow)
    
    # Status do pet
    is_lost = Column(Boolean, default=False)
    lost_date = Column(DateTime, nullable=True)
    found_date = Column(DateTime, nullable=True)
    
    # Localização (opcional)
    last_seen_latitude = Column(Float, nullable=True)
    last_seen_longitude = Column(Float, nullable=True)
    last_seen_address = Column(String(255), nullable=True)

class IdentificationLog(Base):
    __tablename__ = "identification_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Resultado da identificação
    identified_pet_id = Column(String, ForeignKey("pets.id"), nullable=True)
    confidence_score = Column(Float, nullable=False)
    match_found = Column(Boolean, nullable=False)
    
    # Dados da consulta
    query_species = Column(String(20), nullable=False)
    query_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Dados opcionais do usuário que fez a consulta
    query_user_id = Column(String, ForeignKey("users.id"), nullable=True)
    
    # Localização da consulta (opcional)
    query_latitude = Column(Float, nullable=True)
    query_longitude = Column(Float, nullable=True)
    
    # Relacionamentos
    identified_pet = relationship("Pet")
    query_user = relationship("User")

class LostPetReport(Base):
    __tablename__ = "lost_pet_reports"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Pet relacionado
    pet_id = Column(String, ForeignKey("pets.id"), nullable=False)
    pet = relationship("Pet")
    
    # Dados do relato
    reported_by_user_id = Column(String, ForeignKey("users.id"), nullable=False)
    reported_by = relationship("User")
    
    report_date = Column(DateTime, default=datetime.utcnow)
    last_seen_date = Column(DateTime, nullable=True)
    
    # Localização
    last_seen_latitude = Column(Float, nullable=True)
    last_seen_longitude = Column(Float, nullable=True)
    last_seen_address = Column(String(255), nullable=True)
    
    # Descrição adicional
    additional_info = Column(Text, nullable=True)
    
    # Status do relato
    is_resolved = Column(Boolean, default=False)
    resolved_date = Column(DateTime, nullable=True)

class PetImage(Base):
    __tablename__ = "pet_images"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Pet relacionado
    pet_id = Column(String, ForeignKey("pets.id"), nullable=False)
    pet = relationship("Pet", back_populates="images")
    
    # Dados da imagem
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=True)
    mime_type = Column(String(100), nullable=True)
    
    # Dados biométricos específicos desta imagem
    biometric_embedding = Column(LargeBinary, nullable=True)
    biometric_confidence = Column(Float, nullable=True)
    
    # Metadados
    is_primary = Column(Boolean, default=False)  # Imagem principal do pet
    upload_date = Column(DateTime, default=datetime.utcnow)
    uploaded_by_user_id = Column(String, ForeignKey("users.id"), nullable=False)
    uploaded_by = relationship("User")
    
    # Qualidade da imagem para identificação
    quality_score = Column(Float, nullable=True)  # 0.0 a 1.0
    face_detected = Column(Boolean, default=False)
    
    # Status
    is_active = Column(Boolean, default=True)