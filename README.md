# SmartPet ID • Backend (FastAPI)

API para identificação biométrica de pets (face/focinho) com CLIP, autenticação JWT, upload de imagens e integrações de QR/RFID.

## Pré‑requisitos
- Python 3.9+
- Pip
- macOS/Linux/Windows

## Instalação
```bash
cd backend
pip install -r requirements.txt
```

## Variáveis de ambiente
Crie um arquivo `.env` (opcional):
```
SECRET_KEY=troque-este-segredo-em-producao
DATABASE_URL=sqlite:///./pet_biometric.db
```

## Banco de dados
Inicializar tabelas:
```bash
python3 init_db.py
```
ou:
```bash
python3 create_db.py
```

Criar usuário administrador de teste:
```bash
python3 create_admin.py
```
Credenciais exibidas no terminal (padrão: usuário `superadmin`, senha `admin123`).

## Execução
Iniciar servidor (porta 8000):
```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```
Documentação: `http://localhost:8000/docs`

Uploads locais: criados em `uploads/pets/<id>/` (servidos em `http://localhost:8000/static/pets/<id>/<arquivo>`).

## Autenticação
- Registro: `POST /auth/register`
- Login: `POST /auth/login` → retorna `access_token`
- Headers nas rotas protegidas: `Authorization: Bearer <token>`

## Endpoints principais
- Registrar pet (com imagens, autenticado): `POST /pets/register`
- Adicionar imagens: `POST /pets/{pet_id}/add-images`
- Identificar pet por imagem: `POST /pets/identify`
- Listar pets: `GET /pets`
- Detalhes do pet: `GET /pets/{pet_id}`
- QR do pet (conteúdo/deep link): `GET /pets/{pet_id}/qrcode-data`
- RFID do pet (NDEF texto/deep link): `GET /pets/{pet_id}/rfid-data`

### Exemplos (curl)
Identificar pet:
```bash
curl -X POST "http://localhost:8000/pets/identify" \
  -H "Authorization: Bearer <TOKEN>" \
  -F "image=@sua_imagem.jpg" \
  -F "species=auto"
```

Registrar pet (teste, sem auth):
```bash
curl -X POST "http://localhost:8000/pets/register-test" \
  -F "name=Rex" -F "species=dog" -F "owner_contact=test@example.com" \
  -F "image=@imagem_do_pet.jpg"
```

## Observações
- Em desenvolvimento, use `SECRET_KEY` forte e troque em produção.
- Para iOS NFC/RFID no app, é necessário provisionamento com capacidade “Near Field Communication Tag Reading”.
