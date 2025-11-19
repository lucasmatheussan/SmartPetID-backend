from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

class AuthHandler:
    pwd_context = CryptContext(schemes=["bcrypt_sha256", "bcrypt"], deprecated="auto")
    secret_key = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
    algorithm = "HS256"
    access_token_expire_minutes = 30 * 24 * 60  # 30 dias
    
    def get_password_hash(self, password: str) -> str:
        """Gera hash da senha"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verifica se a senha está correta"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def encode_token(self, user_id: str) -> str:
        """Gera token JWT"""
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def decode_token(self, token: str) -> str:
        """Decodifica token JWT e retorna user_id"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id = payload.get("user_id")
            if user_id is None:
                return None
            return user_id
        except JWTError:
            return None
    
    def refresh_token(self, token: str) -> str:
        """Renova token JWT"""
        user_id = self.decode_token(token)
        if user_id:
            return self.encode_token(user_id)
        return None
    
    def generate_reset_token(self, user_id: str) -> str:
        """Gera token para reset de senha (válido por 1 hora)"""
        payload = {
            "user_id": user_id,
            "type": "password_reset",
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_reset_token(self, token: str) -> str:
        """Verifica token de reset e retorna user_id se válido"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            if payload.get("type") != "password_reset":
                return None
            user_id = payload.get("user_id")
            return user_id
        except JWTError:
            return None
    
    def reset_password(self, user_id: str, new_password: str) -> str:
        """Gera hash da nova senha para reset"""
        return self.get_password_hash(new_password)