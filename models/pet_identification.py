import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import clip
import numpy as np
import cv2
from PIL import Image
import faiss
from typing import List, Tuple, Optional, Dict, Any
import pickle
import os
import io
from sqlalchemy.orm import Session

class CLIPPetEncoder(nn.Module):
    """
    CLIP-based pet encoder for MVP - no training required!
    Uses pre-trained CLIP model for high-quality visual embeddings.
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        
        # Load pre-trained CLIP model
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        
        # Freeze CLIP parameters (no training needed)
        for param in self.model.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        with torch.no_grad():
            # Extract CLIP features directly
            features = self.model.encode_image(x)
            features = features.float()
            # CLIP já normaliza os features, mas vamos garantir
            features = F.normalize(features, p=2, dim=1)
            
        return features

# Removed complex models - using CLIP for MVP simplicity
# CLIPPetEncoder above handles both dogs and cats with pre-trained weights

class PetBiometricPreprocessor:
    """
    Preprocessador de imagens para biometria de pets usando CLIP
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # CLIP preprocessing será definido quando carregar o modelo
        self.clip_preprocess = None
        
    def set_clip_preprocess(self, preprocess_fn):
        """Define a função de preprocessamento do CLIP"""
        self.clip_preprocess = preprocess_fn
    
    def detect_and_crop_region(self, image: np.ndarray, species: str) -> Optional[np.ndarray]:
        """Detecta e recorta a região de interesse (focinho para cães, face para gatos)"""
        try:
            if species == "dog":
                # Para cães, focamos na região do focinho
                # Implementação simplificada - em produção usaria detectores específicos
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                
                # Detectar face primeiro
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    # Pegar a maior face detectada
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    
                    # Focar na região inferior da face (focinho)
                    nose_y = y + int(h * 0.6)
                    nose_h = int(h * 0.4)
                    nose_x = x + int(w * 0.2)
                    nose_w = int(w * 0.6)
                    
                    cropped = image[nose_y:nose_y+nose_h, nose_x:nose_x+nose_w]
                    return cropped if cropped.size > 0 else image
                
                return image
            
            elif species == "cat":
                # Para gatos, usamos a face completa
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    cropped = image[y:y+h, x:x+w]
                    return cropped if cropped.size > 0 else image
                
                return image
            
            return image
            
        except Exception as e:
            print(f"Erro na detecção: {e}")
            return image
    
    def preprocess_image(self, image_data: bytes) -> torch.Tensor:
        """
        Preprocessa imagem usando CLIP preprocessing
        """
        # Converter bytes para PIL Image
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Aplicar preprocessamento do CLIP
        if self.clip_preprocess is not None:
            tensor = self.clip_preprocess(image)
        else:
            # Fallback para preprocessamento básico
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            tensor = transform(image)
            
        return tensor.unsqueeze(0).to(self.device)  # Adicionar batch dimension

class PetIdentificationService:
    """Serviço principal para identificação biométrica de pets usando CLIP"""
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.preprocessor = PetBiometricPreprocessor(device)
        
        # Modelo único CLIP para todos os pets
        self.model = None
        
        # Índice FAISS único
        self.index = None
        
        # Metadados dos pets registrados
        self.pet_metadata = []
        
        self._load_model()
    
    def _load_model(self):
        """
        Carrega o modelo CLIP pré-treinado
        """
        try:
            print("Carregando modelo CLIP...")
            self.model = CLIPPetEncoder(device=self.device)
            self.model.eval()
            
            # Configurar preprocessamento
            self.preprocessor.set_clip_preprocess(self.model.preprocess)
            
            print("Modelo CLIP carregado com sucesso!")
                
        except Exception as e:
            print(f"Erro ao carregar modelo CLIP: {e}")
            raise e
    
    async def extract_biometric_features(self, image_data: bytes, species: str) -> Optional[Dict[str, Any]]:
        """
        Extrai características biométricas usando CLIP
        """
        try:
            # Preprocessar imagem
            image_tensor = self.preprocessor.preprocess_image(image_data)
            
            # Extrair features usando CLIP
            with torch.no_grad():
                embedding = self.model(image_tensor)
                embedding_np = embedding.cpu().numpy().flatten()
                
                # Normalizar embedding usando L2 normalization
                norm = np.linalg.norm(embedding_np)
                if norm > 0:
                    embedding_np = embedding_np / norm
            
            # Calcular confidence baseado na qualidade da imagem
            quality_metrics = self.analyze_image_quality(image_data)
            
            # Confidence baseado na qualidade geral da imagem
            base_confidence = 0.7  # Confidence base para CLIP
            quality_bonus = quality_metrics.get('overall', 0.5) * 0.3  # Até 30% de bonus
            confidence = min(base_confidence + quality_bonus, 1.0)
            
            # Embedding extraído com sucesso
            
            return {
                "embedding": embedding_np,
                "embedding_dim": len(embedding_np),
                "confidence": confidence,
                "species": species,
                "model_version": "clip_mvp_v1.0"
            }
            
        except Exception as e:
            print(f"Erro na extração de características: {e}")
            return None
    
    def build_search_index(self, pets_data: List[Dict]):
        """
        Constrói índice FAISS otimizado para múltiplas imagens por pet
        """
        try:
            if not pets_data:
                print("Nenhum pet para indexar")
                return
                
            embeddings = []
            self.pet_metadata = []
            
            for pet in pets_data:
                # Deserializar embedding principal se necessário
                if isinstance(pet['biometric_embedding'], bytes):
                    try:
                        main_embedding = pickle.loads(pet['biometric_embedding'])
                    except Exception as e:
                        print(f"Erro ao deserializar embedding do pet {pet['id']}: {e}")
                        continue
                elif isinstance(pet['biometric_embedding'], np.ndarray):
                    main_embedding = pet['biometric_embedding']
                else:
                    print(f"Tipo de embedding inválido para pet {pet['id']}: {type(pet['biometric_embedding'])}")
                    continue
                
                # Buscar todas as imagens do pet no banco
                from database.models import PetImage
                from database.database import SessionLocal
                
                db = SessionLocal()
                try:
                    pet_images = db.query(PetImage).filter(
                        PetImage.pet_id == pet['id'],
                        PetImage.biometric_embedding.isnot(None)
                    ).all()
                    
                    # Coletar todos os embeddings do pet
                    pet_embeddings = [main_embedding]
                    
                    for img in pet_images:
                        if img.biometric_embedding:
                            if isinstance(img.biometric_embedding, bytes):
                                img_embedding = pickle.loads(img.biometric_embedding)
                            else:
                                img_embedding = img.biometric_embedding
                            pet_embeddings.append(img_embedding)
                    
                    # Estratégia: usar embedding médio ponderado por qualidade
                    if len(pet_embeddings) > 1:
                        # Calcular pesos baseados na confiança
                        weights = [1.0]  # Peso da imagem principal
                        for img in pet_images:
                            if img.biometric_embedding and img.biometric_confidence:
                                weights.append(img.biometric_confidence / 100.0)
                            else:
                                weights.append(0.5)  # Peso padrão
                        
                        # Normalizar pesos
                        weights = np.array(weights)
                        weights = weights / weights.sum()
                        
                        # Calcular embedding médio ponderado
                        weighted_embedding = np.average(pet_embeddings, axis=0, weights=weights)
                        final_embedding = weighted_embedding
                    else:
                        final_embedding = main_embedding
                    
                finally:
                    db.close()
                    
                embeddings.append(final_embedding)
                self.pet_metadata.append({
                    'id': pet['id'],
                    'name': pet['name'],
                    'species': pet['species'],
                    'owner_id': pet['owner_id'],
                    'num_images': len(pet_embeddings)
                })
            
            if embeddings:
                # Criar matriz de embeddings
                embeddings_matrix = np.vstack(embeddings).astype('float32')
                
                # Criar índice FAISS (Inner Product para embeddings normalizados)
                self.index = faiss.IndexFlatIP(embeddings_matrix.shape[1])
                self.index.add(embeddings_matrix)
                
                # Índice construído com sucesso
            
        except Exception as e:
            print(f"Erro ao construir índice: {e}")
    
    async def identify_pet(self, image_data: bytes, species: str, db: Session) -> Dict[str, Any]:
        """
        Identifica um pet usando estratégias avançadas com múltiplas imagens
        """
        try:
            # Extrair características da imagem de consulta
            features = await self.extract_biometric_features(image_data, species)
            if not features:
                return {"success": False, "error": "Falha na extração de características"}
            
            query_embedding = features["embedding"]
            
            # Garantir que o query embedding está normalizado
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
            
            # Query embedding normalizado
            
            # Verificar se há índice construído
            if self.index is None or not self.pet_metadata:
                # Reconstruir índice
                from database.models import Pet
                pets = db.query(Pet).all()
                pets_data = [{
                    'id': pet.id,
                    'name': pet.name,
                    'species': pet.species,
                    'owner_id': pet.owner_id,
                    'biometric_embedding': pet.biometric_embedding
                } for pet in pets]
                
                self.build_search_index(pets_data)
                
                if self.index is None or len(self.pet_metadata) == 0:
                    return {
                        "success": True,
                        "identified": False,
                        "best_match": None,
                        "all_matches": [],
                        "total_registered_pets": 0,
                        "algorithm_used": "clip_mvp_v1.0",
                        "identification_threshold": 0.7,
                        "message": "Nenhum pet válido registrado no sistema para comparação"
                    }
            
            # Estratégia 1: Busca com embedding médio ponderado (índice principal)
            query_embedding = features['embedding']
            query_vector = np.array([query_embedding]).astype('float32')
            
            k = min(10, len(self.pet_metadata))  # Buscar mais candidatos
            scores, indices = self.index.search(query_vector, k)
            
            # Filtrar candidatos por espécie se especificado
            valid_candidates = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.pet_metadata):
                    pet_meta = self.pet_metadata[idx]
                    # Filtrar por espécie se especificado
                    if species and species != 'unknown' and species != 'auto' and pet_meta['species'] != species:
                        continue
                    valid_candidates.append((score, idx, pet_meta))
            
            # Estratégia 2: Comparação individual com todas as imagens dos top candidatos
            enhanced_matches = []
            
            for score, idx, pet_meta in valid_candidates:
                 pet_id = pet_meta['id']
                 
                 # Buscar todas as imagens do pet candidato
                 from database.models import Pet, User, PetImage
                 pet = db.query(Pet).filter(Pet.id == pet_meta['id']).first()
                 
                 if pet:
                        # Coletar todos os embeddings do pet candidato
                        candidate_embeddings = []
                        
                        # Embedding principal
                        if pet.biometric_embedding:
                            if isinstance(pet.biometric_embedding, bytes):
                                main_emb = pickle.loads(pet.biometric_embedding)
                            else:
                                main_emb = pet.biometric_embedding
                            
                            # Normalizar embedding
                            norm = np.linalg.norm(main_emb)
                            if norm > 0:
                                main_emb = main_emb / norm
                            
                            candidate_embeddings.append({
                                'embedding': main_emb,
                                'confidence': pet.biometric_confidence or 1.0,
                                'is_primary': True
                            })
                        
                        # Embeddings das imagens adicionais
                        pet_images = db.query(PetImage).filter(
                            PetImage.pet_id == pet.id,
                            PetImage.biometric_embedding.isnot(None)
                        ).all()
                        
                        for img in pet_images:
                            if img.biometric_embedding:
                                if isinstance(img.biometric_embedding, bytes):
                                    img_emb = pickle.loads(img.biometric_embedding)
                                else:
                                    img_emb = img.biometric_embedding
                                
                                # Normalizar embedding
                                norm = np.linalg.norm(img_emb)
                                if norm > 0:
                                    img_emb = img_emb / norm
                                
                                candidate_embeddings.append({
                                    'embedding': img_emb,
                                    'confidence': img.biometric_confidence or 0.5,
                                    'is_primary': False
                                })
                        
                        # Calcular múltiplas métricas de similaridade
                        similarities = []
                        for emb_data in candidate_embeddings:
                            try:
                                # Verificar compatibilidade de dimensões
                                query_emb = query_embedding
                                stored_emb = emb_data['embedding']
                                
                                # Verificar compatibilidade de dimensões
                                if query_emb.shape != stored_emb.shape:
                                    continue
                                
                                # Similaridade coseno (produto escalar de vetores normalizados)
                                sim = np.dot(query_emb, stored_emb)
                                similarities.append(sim)
                            except Exception as e:
                                continue
                        
                        if similarities:
                            # Estratégias de agregação
                            max_similarity = max(similarities)  # Melhor match individual
                            avg_similarity = np.mean(similarities)  # Média das similaridades
                            
                            # Weighted average usando confidence dos embeddings
                            confidences = [e['confidence'] for e in candidate_embeddings[:len(similarities)]]
                            weighted_avg = np.average(similarities, weights=confidences)
                            
                            # Para imagens idênticas, priorizar o max_similarity
                            if max_similarity > 0.99:  # Praticamente idêntica
                                final_score = max_similarity
                            elif max_similarity > 0.95:  # Muito similar
                                final_score = max_similarity * 0.8 + weighted_avg * 0.2
                            else:  # Similaridade normal
                                final_score = (max_similarity * 0.5 + weighted_avg * 0.3 + avg_similarity * 0.2)
                            
                            # Bonus menor por múltiplas imagens (não deve inflar artificialmente)
                            if len(candidate_embeddings) > 1:
                                multi_image_bonus = min(0.02 * (len(candidate_embeddings) - 1), 0.05)
                                final_score += multi_image_bonus
                            
                            owner = db.query(User).filter(User.id == pet.owner_id).first()
                            
                            enhanced_matches.append({
                                "pet_id": pet.id,
                                "pet_name": pet.name,
                                "species": pet.species,
                                "breed": pet.breed,
                                "similarity_score": float(final_score),
                                "confidence": min(float(final_score * 100), 100.0),
                                "max_individual_similarity": float(max_similarity),
                                "avg_similarity": float(avg_similarity),
                                "num_images_compared": len(candidate_embeddings),
                                "owner_name": owner.full_name if owner else "Desconhecido",
                                "owner_contact": owner.email if owner else None
                            })
            
            # Ordenar por score final
            enhanced_matches.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Pegar apenas os top 5 para retorno
            final_matches = enhanced_matches[:5]
            
            # Determinar identificação positiva com critérios mais rigorosos
            best_match = final_matches[0] if final_matches else None
            
            # Determinar identificação positiva
            if best_match:
                # Critérios adaptativos baseados no número de imagens
                base_threshold = 50.0
                if best_match['num_images_compared'] > 3:
                    threshold = base_threshold - 5.0  # Mais flexível com mais imagens
                elif best_match['num_images_compared'] == 1:
                    threshold = base_threshold + 10.0  # Mais rigoroso com uma imagem
                else:
                    threshold = base_threshold
                
                is_identified = best_match["confidence"] > threshold
            else:
                is_identified = False
                threshold = 50.0
            
            return {
                "success": True,
                "identified": is_identified,
                "best_match": best_match,
                "all_matches": final_matches,
                "total_registered_pets": len(self.pet_metadata),
                "algorithm_used": "multi_image_enhanced",
                "identification_threshold": threshold if best_match else 70.0
            }
            
        except Exception as e:
            import traceback
            print(f"Erro na identificação: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}
    
    def analyze_image_quality(self, image_data: bytes) -> Dict[str, float]:
        """
        Analisa a qualidade da imagem para otimizar a identificação
        """
        try:
            # Converter bytes para imagem
            image = Image.open(io.BytesIO(image_data))
            img_array = np.array(image)
            
            # Métricas de qualidade
            quality_metrics = {}
            
            # 1. Resolução
            width, height = image.size
            resolution_score = min((width * height) / (224 * 224), 1.0)  # Normalizado para 224x224
            quality_metrics['resolution'] = resolution_score
            
            # 2. Nitidez (usando variância do Laplaciano)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalizado
            quality_metrics['sharpness'] = sharpness_score
            
            # 3. Contraste
            contrast = gray.std() / 255.0
            quality_metrics['contrast'] = contrast
            
            # 4. Brilho (evitar imagens muito escuras ou claras)
            brightness = gray.mean() / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Melhor em torno de 0.5
            quality_metrics['brightness'] = brightness_score
            
            # Score geral
            overall_quality = (
                resolution_score * 0.2 +
                sharpness_score * 0.4 +
                contrast * 0.2 +
                brightness_score * 0.2
            )
            quality_metrics['overall'] = overall_quality
            
            return quality_metrics
            
        except Exception as e:
            print(f"Erro na análise de qualidade: {e}")
            return {'overall': 0.5, 'resolution': 0.5, 'sharpness': 0.5, 'contrast': 0.5, 'brightness': 0.5}
    
    def get_identification_strategy_recommendation(self, db: Session, pet_id: str = None) -> Dict[str, Any]:
        """
        Recomenda a melhor estratégia de identificação baseada nas imagens disponíveis
        """
        try:
            if pet_id:
                # Análise para um pet específico
                from database.models import Pet, PetImage
                
                pet = db.query(Pet).filter(Pet.id == pet_id).first()
                if not pet:
                    return {"error": "Pet não encontrado"}
                
                pet_images = db.query(PetImage).filter(PetImage.pet_id == pet_id).all()
                
                total_images = len(pet_images) + (1 if pet.biometric_embedding else 0)
                high_quality_images = sum(1 for img in pet_images if (img.quality_score or 0) > 0.7)
                
                if pet.biometric_confidence and pet.biometric_confidence > 0.7:
                    high_quality_images += 1
                
                recommendation = {
                    "pet_id": pet_id,
                    "pet_name": pet.name,
                    "total_images": total_images,
                    "high_quality_images": high_quality_images,
                    "identification_confidence": "high" if high_quality_images >= 3 else "medium" if high_quality_images >= 2 else "low",
                    "recommended_action": ""
                }
                
                if total_images < 3:
                    recommendation["recommended_action"] = "Adicionar mais imagens para melhorar precisão"
                elif high_quality_images < 2:
                    recommendation["recommended_action"] = "Adicionar imagens de melhor qualidade"
                else:
                    recommendation["recommended_action"] = "Configuração ótima para identificação"
                
                return recommendation
            
            else:
                # Análise geral do sistema
                from database.models import Pet, PetImage
                
                total_pets = db.query(Pet).count()
                pets_with_multiple_images = db.query(Pet).join(PetImage).group_by(Pet.id).having(db.func.count(PetImage.id) > 1).count()
                
                avg_images_per_pet = db.query(db.func.avg(db.func.count(PetImage.id))).select_from(PetImage).group_by(PetImage.pet_id).scalar() or 1
                
                return {
                    "total_pets": total_pets,
                    "pets_with_multiple_images": pets_with_multiple_images,
                    "avg_images_per_pet": float(avg_images_per_pet),
                    "system_optimization": "high" if avg_images_per_pet > 2 else "medium" if avg_images_per_pet > 1.5 else "low"
                }
                
        except Exception as e:
            return {"error": str(e)}
    
    def add_pet_to_index(self, pet_data: Dict):
        """
        Adiciona um novo pet ao índice existente com suporte a múltiplas imagens
        """
        try:
            if self.index is None:
                # Se não há índice, criar um novo
                self.build_search_index([pet_data])
                return
            
            # Reconstruir índice para incluir múltiplas imagens do novo pet
            # (Mais eficiente que tentar adicionar incrementalmente)
            from database.models import Pet
            from database.database import SessionLocal
            
            db = SessionLocal()
            try:
                pets = db.query(Pet).all()
                pets_data = [{
                    'id': pet.id,
                    'name': pet.name,
                    'species': pet.species,
                    'owner_id': pet.owner_id,
                    'biometric_embedding': pet.biometric_embedding
                } for pet in pets]
                
                self.build_search_index(pets_data)
                print(f"Índice reconstruído incluindo pet {pet_data.get('name', 'desconhecido')}")
                
            finally:
                db.close()
            
        except Exception as e:
            print(f"Erro ao adicionar pet ao índice: {e}")