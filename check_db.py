import sqlite3

conn = sqlite3.connect('pet_biometric.db')
cursor = conn.cursor()

# Verificar estrutura da tabela pet_images
cursor.execute('PRAGMA table_info(pet_images)')
columns = cursor.fetchall()
print('Colunas da tabela pet_images:')
for col in columns:
    print(f'  {col[1]} ({col[2]})')

# Verificar algumas imagens
cursor.execute('SELECT * FROM pet_images LIMIT 3')
images = cursor.fetchall()
print('\nPrimeiras 3 imagens:')
for img in images:
    print(f'  {img}')

conn.close()