import ollama

# Pormpt de base pour l'exemple
prompt = "J'adore manger du fromage à raclette."

# Récupère les embeddings pour le prompt
embeddings = ollama.embeddings(
  model='nomic-embed-text',
  prompt=prompt,
)

# Affichage du résultat de l'embedding pour le prompt
print(f"Embeddings for '{prompt}':")
print(embeddings.embedding)