# SmartPet ID Biometric Identification MVP

Sistema de identificação biométrica de pets usando CLIP (Computer Vision) para reconhecimento facial/focinho.

## 🚀 Setup Rápido

### 1. Instalar Dependências
```bash
cd backend
pip install -r requirements.txt
```

### 2. Inicializar Banco de Dados
```bash
python3 create_db.py
```

### 3. Iniciar Servidor
```bash
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

### 4. Acessar API
- **Documentação**: http://localhost:8001/docs
- **API Base**: http://localhost:8001

## 📋 Endpoints Principais

### Identificação de Pet
```bash
curl -X POST "http://localhost:8001/pets/identify" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@sua_imagem.jpg" \
  -F "species=auto"
```

### Registro de Pet (Teste)
```bash
curl -X POST "http://localhost:8001/pets/register-test" \
  -H "Content-Type: multipart/form-data" \
  -F "name=Rex" \
  -F "species=dog" \
  -F "breed=Golden Retriever" \
  -F "age=3" \
  -F "description=Cachorro amigável" \
  -F "owner_contact=test@example.com" \
  -F "image=@imagem_do_pet.jpg"
```

## 🔧 Tecnologias

- **Backend**: FastAPI + Python 3.9+
- **IA**: OpenAI CLIP (ViT-B/32)
- **Banco**: SQLite (desenvolvimento)
- **Embeddings**: 512 dimensões (CLIP nativo)

## 📊 Status do Sistema

✅ **Funcionando**:
- Extração de embeddings com CLIP
- Identificação de pets (retorna 200 OK)
- Banco de dados SQLite
- API endpoints básicos

⚠️ **Limitações Atuais**:
- Endpoint de registro com autenticação tem problemas
- Usar `/pets/register-test` para testes
- Banco vazio inicialmente (sem pets pré-cadastrados)

## 🧪 Teste Rápido

1. Coloque uma imagem de teste como `test_image.png` no diretório backend
2. Execute o comando de identificação acima
3. Deve retornar: `{"match_found": false}` (normal, banco vazio)

## 📝 Próximos Passos

1. Corrigir endpoint de registro com autenticação
2. Adicionar pets de exemplo no banco
3. Implementar interface Flutter
4. Otimizar threshold de similaridade
5. Deploy em produção
