from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os
import logging
import traceback
from datetime import datetime
from sqlalchemy.orm import Session

from models.pet_identification import PetIdentificationService
from database.database import get_db, create_tables
from database.models import Pet, User, PetImage
from auth.auth_handler import AuthHandler

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pet Biometric Identification API",
    description="Sistema de identificação biométrica de pets baseado em focinho/face",
    version="1.0.0"
)

# Configurar arquivos estáticos
os.makedirs("uploads/pets", exist_ok=True)
app.mount("/static", StaticFiles(directory="uploads"), name="static")

# Middleware para capturar exceções
class ExceptionLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Unhandled exception in {request.method} {request.url}:")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception message: {str(e)}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise e

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar domínios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Adicionar middleware de logging de exceções
app.add_middleware(ExceptionLoggingMiddleware)

# Criar tabelas se não existirem
create_tables()

# Instâncias dos serviços
auth_handler = AuthHandler()
pet_service = PetIdentificationService()
security = HTTPBearer()

# Modelos Pydantic
class PetRegistration(BaseModel):
    name: str
    species: str  # "dog" ou "cat"
    breed: Optional[str] = None
    age: Optional[int] = None
    description: Optional[str] = None
    owner_contact: str

class UserRegistration(BaseModel):
    username: str
    email: str
    password: str
    full_name: str
    phone: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class PasswordResetRequest(BaseModel):
    email: str

class PasswordReset(BaseModel):
    token: str
    new_password: str

class IdentificationResult(BaseModel):
    pet_id: Optional[str]
    confidence: float  # Confiança em porcentagem (0-100)
    pet_name: Optional[str]
    owner_contact: Optional[str]
    breed: Optional[str] = None
    age: Optional[str] = None
    description: Optional[str] = None
    last_seen: Optional[str] = None
    status: Optional[str] = None
    match_found: bool
    images: Optional[List[dict]] = None  # URLs das imagens do pet

class QrCodeData(BaseModel):
    qr_content: str
    deep_link_url: str
    pet_id: str
    pet_name: Optional[str] = None

class RfidData(BaseModel):
    ndef_text: str
    deep_link_url: str
    pet_id: str
    pet_name: Optional[str] = None

# Dependência de autenticação
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    user_id = auth_handler.decode_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Token inválido")
    return user_id

@app.get("/")
async def root():
    return {"message": "Pet Biometric Identification API", "status": "running"}

@app.post("/auth/register")
async def register_user(user_data: UserRegistration, db=Depends(get_db)):
    """Registra um novo usuário"""
    try:
        # Verificar se usuário já existe
        existing_user = db.query(User).filter(
            (User.username == user_data.username) | (User.email == user_data.email)
        ).first()
        
        if existing_user:
            raise HTTPException(status_code=400, detail="Usuário ou email já existe")
        
        # Criar novo usuário
        hashed_password = auth_handler.get_password_hash(user_data.password)
        new_user = User(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password,
            full_name=user_data.full_name,
            phone=user_data.phone
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        return {"message": "Usuário criado com sucesso", "user_id": new_user.id}
    
    except Exception as e:
        import traceback
        import logging
        
        # Configurar logging para garantir que apareça
        logging.basicConfig(level=logging.ERROR)
        logger = logging.getLogger(__name__)
        
        error_msg = f"Erro no registro de pet: {e}"
        error_type = f"Tipo do erro: {type(e)}"
        error_trace = f"Traceback completo: {traceback.format_exc()}"
        
        # Usar tanto print quanto logging
        print("=" * 50)
        print(error_msg)
        print(error_type)
        print(error_trace)
        print("=" * 50)
        
        logger.error("=" * 50)
        logger.error(error_msg)
        logger.error(error_type)
        logger.error(error_trace)
        logger.error("=" * 50)
        
        try:
            db.rollback()  # Rollback em caso de erro
        except:
            pass
            
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)} - Traceback: {traceback.format_exc()}")

@app.post("/auth/login")
async def login_user(user_credentials: UserLogin, db=Depends(get_db)):
    """Autentica um usuário"""
    try:
        user = db.query(User).filter(User.username == user_credentials.username).first()
        
        if not user or not auth_handler.verify_password(user_credentials.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Credenciais inválidas")
        
        token = auth_handler.encode_token(user.id)
        return {"access_token": token, "token_type": "bearer", "user_id": user.id}
    
    except Exception as e:
        logger.error(f"=== ERRO NA IDENTIFICAÇÃO ===")
        logger.error(f"Tipo do erro: {type(e)}")
        logger.error(f"Mensagem: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.error(f"=== FIM ERRO ===")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/request-password-reset")
async def request_password_reset(reset_request: PasswordResetRequest, db=Depends(get_db)):
    """Solicita reset de senha por email"""
    try:
        user = db.query(User).filter(User.email == reset_request.email).first()
        
        if not user:
            # Por segurança, não revelamos se o email existe ou não
            return {"message": "Se o email existir, um token de reset será enviado"}
        
        reset_token = auth_handler.generate_reset_token(user.id)
        
        # Em produção, aqui você enviaria o token por email
        # Por enquanto, retornamos o token diretamente (apenas para desenvolvimento)
        return {
            "message": "Token de reset gerado com sucesso",
            "reset_token": reset_token,  # Remover em produção
            "note": "Em produção, este token seria enviado por email"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pets/{pet_id}/qrcode-data", response_model=QrCodeData)
async def get_pet_qrcode_data(
    pet_id: str,
    request: Request,
    current_user: str = Depends(get_current_user),
    db=Depends(get_db)
):
    try:
        pet = db.query(Pet).filter(Pet.id == pet_id).first()
        if not pet:
            raise HTTPException(status_code=404, detail="Pet não encontrado")

        base_url = str(request.base_url)
        if not base_url.endswith("/"):
            base_url += "/"

        deep_link_url = f"{base_url}pets/{pet_id}"
        qr_content = f"focinhoid:pet:{pet_id}"

        return QrCodeData(
            qr_content=qr_content,
            deep_link_url=deep_link_url,
            pet_id=str(pet.id),
            pet_name=pet.name,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pets/{pet_id}/rfid-data", response_model=RfidData)
async def get_pet_rfid_data(
    pet_id: str,
    request: Request,
    current_user: str = Depends(get_current_user),
    db=Depends(get_db)
):
    try:
        pet = db.query(Pet).filter(Pet.id == pet_id).first()
        if not pet:
            raise HTTPException(status_code=404, detail="Pet não encontrado")

        base_url = str(request.base_url)
        if not base_url.endswith("/"):
            base_url += "/"

        deep_link_url = f"{base_url}pets/{pet_id}"
        ndef_text = f"focinhoid:pet:{pet_id}"

        return RfidData(
            ndef_text=ndef_text,
            deep_link_url=deep_link_url,
            pet_id=str(pet.id),
            pet_name=pet.name,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/reset-password")
async def reset_password(reset_data: PasswordReset, db=Depends(get_db)):
    """Reseta a senha usando o token"""
    try:
        # Verifica se o token é válido
        user_id = auth_handler.verify_reset_token(reset_data.token)
        
        if not user_id:
            raise HTTPException(status_code=400, detail="Token inválido ou expirado")
        
        # Busca o usuário
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(status_code=404, detail="Usuário não encontrado")
        
        # Atualiza a senha
        new_hashed_password = auth_handler.reset_password(user_id, reset_data.new_password)
        user.hashed_password = new_hashed_password
        
        db.commit()
        
        return {"message": "Senha resetada com sucesso"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/auth/me")
async def get_current_user_info(current_user: str = Depends(get_current_user), db=Depends(get_db)):
    """Retorna informações do usuário atual baseado no token"""
    try:
        user = db.query(User).filter(User.id == current_user).first()
        
        if not user:
            raise HTTPException(status_code=404, detail="Usuário não encontrado")
        
        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "phone": user.phone
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pets/register", response_model=dict)
async def register_pet(
    name: str = Form(...),
    species: str = Form(...),
    breed: Optional[str] = Form(None),
    age: Optional[int] = Form(None),
    description: Optional[str] = Form(None),
    owner_contact: str = Form(...),
    images: List[UploadFile] = File(...),
    current_user: str = Depends(get_current_user),
    db=Depends(get_db)
):
    """Registra um novo pet com múltiplas imagens biométricas"""
    logger.info(f"=== INÍCIO DO REGISTRO DE PET ===\nUsuário: {current_user}\nNome: {name}\nEspécie: {species}")
    try:
        logger.info("Validando imagens...")
        # Validar que pelo menos uma imagem foi enviada
        if not images or len(images) == 0:
            raise HTTPException(status_code=400, detail="Pelo menos uma imagem é obrigatória")
        
        logger.info(f"Número de imagens recebidas: {len(images)}")
        # Validar tipos de arquivo
        valid_content_types = ['image/', 'application/octet-stream']
        for img in images:
            logger.info(f"Arquivo: {img.filename}, Content-Type: {img.content_type}")
            if img.content_type:
                is_valid_type = any(img.content_type.startswith(ct) for ct in valid_content_types)
                if not is_valid_type:
                    raise HTTPException(status_code=400, detail=f"Arquivo {img.filename} deve ser uma imagem (tipo: {img.content_type})")
                logger.info(f"Tipo de arquivo aceito: {img.content_type}")
            else:
                logger.warning(f"Content-Type não definido para {img.filename}, assumindo que é imagem")
        
        logger.info("Processando primeira imagem...")
        # Processar primeira imagem para criar o embedding principal
        first_image_data = await images[0].read()
        logger.info(f"Dados da imagem lidos: {len(first_image_data)} bytes")
        
        logger.info("Extraindo características biométricas...")
        main_biometric_data = await pet_service.extract_biometric_features(
            first_image_data, species
        )
        logger.info("Características extraídas com sucesso")
        
        if not main_biometric_data:
            raise HTTPException(
                status_code=400, 
                detail="Não foi possível extrair características biométricas da primeira imagem"
            )
        
        # Criar pet no banco de dados
        logger.info("Preparando dados do pet para o banco...")
        import pickle
        logger.info("Serializando embedding...")
        embedding_data = pickle.dumps(main_biometric_data['embedding'])
        logger.info(f"Embedding serializado: {len(embedding_data)} bytes")
        
        logger.info("Criando objeto Pet...")
        new_pet = Pet(
            name=name,
            species=species,
            breed=breed,
            age=age,
            description=description,
            owner_contact=owner_contact,
            owner_id=current_user,
            biometric_embedding=embedding_data,
            biometric_confidence=main_biometric_data['confidence'],
            registration_date=datetime.utcnow()
        )
        
        logger.info("Adicionando pet ao banco...")
        db.add(new_pet)
        logger.info("Fazendo commit...")
        db.commit()
        logger.info("Fazendo refresh...")
        db.refresh(new_pet)
        logger.info(f"Pet criado com ID: {new_pet.id}")
        
        # Processar e salvar todas as imagens
        logger.info("Processando e salvando imagens...")
        saved_images = []
        
        # Criar diretório para o pet se não existir
        pet_dir = f"uploads/pets/{new_pet.id}"
        os.makedirs(pet_dir, exist_ok=True)
        
        for i, img in enumerate(images):
            logger.info(f"Processando imagem {i+1}/{len(images)}")
            # Reset file pointer
            await img.seek(0)
            img_data = await img.read()
            
            # Extrair características biométricas de cada imagem
            img_biometric = await pet_service.extract_biometric_features(
                img_data, species
            )
            
            # Salvar arquivo físico no disco
            filename = img.filename or f"image_{i+1}.jpg"
            file_path = f"{pet_dir}/{filename}"
            
            with open(file_path, "wb") as f:
                f.write(img_data)
            
            logger.info(f"Imagem salva em: {file_path}")
            
            # Criar registro da imagem
            pet_image = PetImage(
                pet_id=new_pet.id,
                filename=filename,
                file_path=file_path,
                file_size=len(img_data),
                mime_type=img.content_type,
                biometric_embedding=pickle.dumps(img_biometric['embedding']) if img_biometric else None,
                biometric_confidence=img_biometric['confidence'] if img_biometric else None,
                is_primary=(i == 0),  # Primeira imagem é a principal
                uploaded_by_user_id=current_user,
                face_detected=bool(img_biometric),
                quality_score=img_biometric['confidence'] if img_biometric else 0.0
            )
            
            db.add(pet_image)
            saved_images.append({
                "filename": pet_image.filename,
                "confidence": pet_image.biometric_confidence,
                "is_primary": pet_image.is_primary
            })
        
        db.commit()
        
        return {
            "message": "Pet registrado com sucesso",
            "pet_id": new_pet.id,
            "main_confidence": main_biometric_data['confidence'],
            "images_processed": len(images),
            "images_details": saved_images
        }
    
    except Exception as e:
        logger.error(f"Erro no registro de pet: {type(e).__name__}: {str(e)}")
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: {str(e)}")

@app.post("/pets/{pet_id}/add-images")
async def add_pet_images(
    pet_id: str,
    images: List[UploadFile] = File(...),
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Adiciona novas imagens a um pet já registrado"""
    try:
        # Verificar se o pet existe e pertence ao usuário
        pet = db.query(Pet).filter(
            Pet.id == pet_id,
            Pet.owner_id == current_user
        ).first()
        
        if not pet:
            raise HTTPException(
                status_code=404, 
                detail="Pet não encontrado ou você não tem permissão para modificá-lo"
            )
        
        # Validar que pelo menos uma imagem foi enviada
        if not images or len(images) == 0:
            raise HTTPException(status_code=400, detail="Pelo menos uma imagem é obrigatória")
        
        # Validar tipos de arquivo
        valid_content_types = ['image/', 'application/octet-stream']
        for img in images:
            if img.content_type:
                is_valid_type = any(img.content_type.startswith(ct) for ct in valid_content_types)
                if not is_valid_type:
                    raise HTTPException(status_code=400, detail=f"Arquivo {img.filename} deve ser uma imagem (tipo: {img.content_type})")
            else:
                # Se não há content_type, assumir que é uma imagem válida
                pass
        
        # Processar e salvar todas as novas imagens
        saved_images = []
        for i, img in enumerate(images):
            img_data = await img.read()
            
            # Extrair características biométricas de cada imagem
            img_biometric = await pet_service.extract_biometric_features(
                img_data, pet.species
            )
            
            # Criar registro da imagem
            pet_image = PetImage(
                pet_id=pet.id,
                filename=img.filename or f"additional_image_{i+1}.jpg",
                file_path=f"uploads/pets/{pet.id}/{img.filename or f'additional_image_{i+1}.jpg'}",
                file_size=len(img_data),
                mime_type=img.content_type,
                biometric_embedding=img_biometric['embedding'] if img_biometric else None,
                biometric_confidence=img_biometric['confidence'] if img_biometric else None,
                is_primary=False,  # Imagens adicionais nunca são primárias
                uploaded_by_user_id=current_user,
                face_detected=bool(img_biometric),
                quality_score=img_biometric['confidence'] if img_biometric else 0.0
            )
            
            db.add(pet_image)
            saved_images.append({
                "filename": pet_image.filename,
                "confidence": pet_image.biometric_confidence,
                "face_detected": pet_image.face_detected
            })
        
        db.commit()
        
        # Contar total de imagens do pet
        total_images = db.query(PetImage).filter(PetImage.pet_id == pet.id).count()
        
        return {
            "message": "Imagens adicionadas com sucesso",
            "pet_id": pet.id,
            "pet_name": pet.name,
            "new_images_added": len(images),
            "total_images": total_images,
            "images_details": saved_images
        }
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pets/{pet_id}/analysis")
async def analyze_pet_identification(
    pet_id: str,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Analisa a qualidade das imagens e estratégia de identificação de um pet"""
    try:
        # Verificar se o pet existe e pertence ao usuário
        pet = db.query(Pet).filter(
            Pet.id == pet_id,
            Pet.owner_id == current_user
        ).first()
        
        if not pet:
            raise HTTPException(
                status_code=404, 
                detail="Pet não encontrado ou você não tem permissão para analisá-lo"
            )
        
        # Obter recomendações do serviço de identificação
        recommendations = pet_service.get_identification_strategy_recommendation(db, pet_id)
        
        # Buscar detalhes das imagens
        pet_images = db.query(PetImage).filter(PetImage.pet_id == pet_id).all()
        
        images_details = []
        for img in pet_images:
            images_details.append({
                "filename": img.filename,
                "quality_score": img.quality_score,
                "biometric_confidence": img.biometric_confidence,
                "is_primary": img.is_primary,
                "face_detected": img.face_detected,
                "upload_date": img.upload_date.isoformat() if img.upload_date else None
            })
        
        return {
            "pet_info": {
                "id": pet.id,
                "name": pet.name,
                "species": pet.species,
                "breed": pet.breed
            },
            "analysis": recommendations,
            "images_details": images_details,
            "optimization_tips": [
                "Use imagens com boa iluminação",
                "Foque no rosto/focinho do animal",
                "Evite imagens borradas ou de baixa resolução",
                "Adicione pelo menos 3 imagens de ângulos diferentes",
                "Mantenha o animal calmo durante a foto"
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system/analysis")
async def analyze_system_performance(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Analisa o desempenho geral do sistema de identificação"""
    try:
        # Obter análise geral do sistema
        system_analysis = pet_service.get_identification_strategy_recommendation(db)
        
        # Estatísticas adicionais
        total_images = db.query(PetImage).count()
        high_quality_images = db.query(PetImage).filter(PetImage.quality_score > 0.7).count()
        
        return {
            "system_analysis": system_analysis,
            "statistics": {
                "total_images": total_images,
                "high_quality_images": high_quality_images,
                "quality_ratio": high_quality_images / total_images if total_images > 0 else 0
            },
            "recommendations": {
                "optimal_images_per_pet": 3,
                "minimum_quality_score": 0.7,
                "recommended_angles": ["frontal", "perfil_esquerdo", "perfil_direito"]
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pets/register-test")
async def register_pet_test(
    name: str = Form(...),
    species: str = Form(...),
    breed: Optional[str] = Form(None),
    age: Optional[int] = Form(None),
    description: Optional[str] = Form(None),
    owner_contact: str = Form(...),
    image: UploadFile = File(...),
    db=Depends(get_db)
):
    """Endpoint temporário para registrar pets sem autenticação"""
    import pickle
    import numpy as np
    
    logger.info("=== INÍCIO DO REGISTRO DE PET (TESTE) ===")
    
    # Validar tipo de arquivo
    valid_content_types = ['image/', 'application/octet-stream']
    if image.content_type:
        is_valid_type = any(image.content_type.startswith(ct) for ct in valid_content_types)
        if not is_valid_type:
            raise HTTPException(status_code=400, detail=f"Arquivo deve ser uma imagem (tipo: {image.content_type})")
    # Se não há content_type, assumir que é uma imagem válida
    
    logger.info("Validação de arquivo OK")
    
    # Processar imagem
    image_data = await image.read()
    logger.info("Leitura da imagem OK")
    
    biometric_data = await pet_service.extract_biometric_features(
        image_data, species
    )
    logger.info("Extração de características OK")
    
    if not biometric_data:
        raise HTTPException(
            status_code=400, 
            detail="Não foi possível extrair características biométricas"
        )
    
    logger.info("Validação de características OK")
    
    # Criar usuário de teste se não existir
    logger.info("Verificando usuário de teste...")
    test_user = db.query(User).filter(User.email == "test@example.com").first()
    if not test_user:
        logger.info("Criando usuário de teste...")
        test_user = User(
            username="test_user",
            email="test@example.com",
            hashed_password="dummy_hash",
            full_name="Test User"
        )
        db.add(test_user)
        db.commit()
        db.refresh(test_user)
        logger.info("Usuário criado com sucesso")
    else:
        logger.info("Usuário já existe")
    
    logger.info("Preparando embedding...")
    # Garantir que o embedding seja um numpy array antes de serializar
    embedding = biometric_data['embedding']
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)
    
    logger.info(f"Embedding preparado - Shape: {embedding.shape}, Type: {type(embedding)}")
    
    # Serializar embedding
    serialized_embedding = pickle.dumps(embedding.astype(np.float32))
    logger.info(f"Embedding serializado - Size: {len(serialized_embedding)} bytes")
    
    logger.info("Criando pet no banco...")
    new_pet = Pet(
        name=name,
        species=species,
        breed=breed,
        age=age,
        description=description,
        owner_contact=owner_contact,
        owner_id=test_user.id,
        biometric_embedding=serialized_embedding,
        biometric_confidence=biometric_data['confidence'],
        registration_date=datetime.utcnow()
    )
    
    logger.info("Adicionando ao banco...")
    db.add(new_pet)
    logger.info("Fazendo commit...")
    db.commit()
    logger.info("Fazendo refresh...")
    db.refresh(new_pet)
    
    # Criar diretório para o pet e salvar a imagem física
    pet_dir = f"uploads/pets/{new_pet.id}"
    os.makedirs(pet_dir, exist_ok=True)
    
    # Reset file pointer e salvar imagem
    await image.seek(0)
    image_data_for_save = await image.read()
    
    filename = image.filename or f"image_1.jpg"
    file_path = f"{pet_dir}/{filename}"
    
    logger.info(f"Salvando imagem física em: {file_path}")
    with open(file_path, "wb") as f:
        f.write(image_data_for_save)
    
    # Criar registro da imagem no banco
    pet_image = PetImage(
        pet_id=new_pet.id,
        filename=filename,
        file_path=file_path,
        file_size=len(image_data_for_save),
        mime_type=image.content_type,
        biometric_embedding=serialized_embedding,
        biometric_confidence=biometric_data['confidence'],
        is_primary=True,
        uploaded_by_user_id=test_user.id,
        face_detected=True,
        quality_score=biometric_data['confidence']
    )
    
    db.add(pet_image)
    db.commit()
    
    logger.info("=== REGISTRO CONCLUÍDO COM SUCESSO ===")
    
    return {
        "success": True,
        "message": "Pet registrado com sucesso",
        "pet_id": new_pet.id,
        "confidence": biometric_data['confidence'],
        "image_saved": file_path
    }



@app.post("/pets/identify")
async def identify_pet(
    image: UploadFile = File(...),
    species: str = Form("auto"),  # "dog", "cat" ou "auto"
    current_user: str = Depends(get_current_user),
    db=Depends(get_db)
):
    """Identifica um pet através de sua imagem biométrica"""
    try:
        logger.info(f"=== INÍCIO IDENTIFICAÇÃO PET - User: {current_user} ===")
        logger.info(f"Arquivo: {image.filename}, Tipo: {image.content_type}, Species: {species}")
        
        # Validar tipo de arquivo
        valid_content_types = ['image/', 'application/octet-stream']
        is_valid_type = any(image.content_type.startswith(ct) for ct in valid_content_types)
        
        if not is_valid_type:
            logger.error(f"Tipo de arquivo inválido: {image.content_type}")
            raise HTTPException(status_code=400, detail="Arquivo deve ser uma imagem")
        
        logger.info(f"Tipo de arquivo aceito: {image.content_type}")
        
        # Ler imagem
        logger.info("Lendo dados da imagem...")
        image_data = await image.read()
        logger.info(f"Imagem lida: {len(image_data)} bytes")
        
        # Identificar pet
        logger.info("Iniciando identificação no serviço...")
        service_result = await pet_service.identify_pet(
            image_data, species, db
        )
        logger.info(f"Resultado do serviço: {service_result}")
        
        # Verificar se houve erro no serviço
        if not service_result.get("success", False):
            raise HTTPException(status_code=500, detail=service_result.get("error", "Erro na identificação"))
        
        # Converter estrutura do serviço para formato esperado pelo app
        best_match = service_result.get("best_match")
        is_identified = service_result.get("identified", False)
        
        if is_identified and best_match:
            # Buscar imagens do pet
            pet_images = db.query(PetImage).filter(PetImage.pet_id == best_match["pet_id"]).all()
            images_list = []
            for img in pet_images:
                images_list.append({
                    "id": img.id,
                    "filename": img.filename,
                    "url": f"http://192.168.1.184:8000/static/pets/{best_match['pet_id']}/{img.filename}",
                    "is_primary": img.is_primary,
                    "quality_score": img.quality_score
                })
            
            result_data = {
                "pet_id": str(best_match["pet_id"]),
                "confidence": round(best_match["confidence"], 1),
                "pet_name": best_match["pet_name"],
                "owner_contact": best_match["owner_contact"],
                "breed": best_match.get("breed"),
                "age": best_match.get("age"),
                "description": None,
                "last_seen": None,
                "status": "found",
                "match_found": True,
                "images": images_list
            }
        else:
            result_data = {
                "pet_id": None,
                "confidence": 0.0,
                "pet_name": None,
                "owner_contact": None,
                "breed": None,
                "age": None,
                "description": None,
                "last_seen": None,
                "status": None,
                "match_found": False,
                "images": None
            }
        
        return result_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pets/user/{user_id}")
async def get_user_pets(
    user_id: str,
    current_user: str = Depends(get_current_user),
    db=Depends(get_db)
):
    """Retorna todos os pets de um usuário"""
    try:
        # Verificar se o usuário pode acessar estes pets
        if current_user != user_id:
            raise HTTPException(status_code=403, detail="Acesso negado")
        
        pets = db.query(Pet).filter(Pet.owner_id == user_id).all()
        
        return {
            "pets": [
                {
                    "id": pet.id,
                    "name": pet.name,
                    "species": pet.species,
                    "breed": pet.breed,
                    "age": pet.age,
                    "description": pet.description,
                    "registration_date": pet.registration_date
                }
                for pet in pets
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pets")
async def get_all_pets(
    current_user: str = Depends(get_current_user),
    db=Depends(get_db)
):
    """Retorna lista de todos os pets cadastrados"""
    try:
        pets = db.query(Pet).all()
        
        pets_with_images = []
        for pet in pets:
            # Buscar imagens do pet
            images = db.query(PetImage).filter(PetImage.pet_id == pet.id).all()
            
            pet_data = {
                "id": pet.id,
                "name": pet.name,
                "species": pet.species,
                "breed": pet.breed,
                "age": pet.age,
                "description": pet.description,
                "owner_contact": pet.owner_contact,
                "registration_date": pet.registration_date,
                "owner_id": pet.owner_id,
                "images": [
                    {
                        "id": img.id,
                        "filename": img.filename,
                        "file_path": img.file_path,
                        "quality_score": img.quality_score,
                        "is_primary": img.is_primary,
                        "url": f"http://192.168.1.184:8000/static/pets/{pet.id}/{img.filename}"
                    }
                    for img in images
                ]
            }
            pets_with_images.append(pet_data)
        
        return {
            "pets": pets_with_images
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pets/{pet_id}")
async def get_pet_details(
    pet_id: str,
    current_user: str = Depends(get_current_user),
    db=Depends(get_db)
):
    """Retorna detalhes de um pet específico com suas imagens"""
    try:
        pet = db.query(Pet).filter(Pet.id == pet_id).first()
        
        if not pet:
            raise HTTPException(status_code=404, detail="Pet não encontrado")
        
        # Buscar imagens do pet
        images = db.query(PetImage).filter(PetImage.pet_id == pet.id).all()
        
        # Construir lista de imagens
        images_list = []
        for img in images:
            images_list.append({
                "id": img.id,
                "filename": img.filename,
                "file_path": img.file_path,
                "quality_score": img.quality_score,
                "is_primary": img.is_primary,
                "url": f"http://192.168.1.184:8000/static/pets/{pet.id}/{img.filename}"
            })
        
        return {
            "id": pet.id,
            "name": pet.name,
            "species": pet.species,
            "breed": pet.breed,
            "age": pet.age,
            "description": pet.description,
            "owner_contact": pet.owner_contact,
            "registration_date": pet.registration_date,
            "owner_id": pet.owner_id,
            "images": images_list
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pets/{pet_id}/{filename}")
async def serve_pet_image(pet_id: str, filename: str):
    """Serve imagens dos pets"""
    from fastapi.responses import FileResponse
    import os
    
    file_path = f"uploads/pets/{pet_id}/{filename}"
    
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="Imagem não encontrada")

@app.get("/pets/lost")
async def get_lost_pets(db=Depends(get_db)):
    """Retorna lista de pets perdidos"""
    try:
        # Em uma implementação real, haveria um campo 'is_lost' na tabela Pet
        lost_pets = db.query(Pet).filter(Pet.is_lost == True).all()
        
        return {
            "lost_pets": [
                {
                    "id": pet.id,
                    "name": pet.name,
                    "species": pet.species,
                    "breed": pet.breed,
                    "description": pet.description,
                    "owner_contact": pet.owner_contact,
                    "lost_date": pet.lost_date
                }
                for pet in lost_pets
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)