import tiktoken

# Nom de l'encodage
#    `o200k_base` : `gpt-4o`, `gpt-4o-mini`
#    `cl100k_base` : `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo`, `text-embedding-ada-002`, `text-embedding-3-small`, `text-embedding-3-large`
#    `p50k_base` : Codex models, `text-davinci-002`, `text-davinci-003`
#    `r50k_base` (or `gpt2`) : GPT-3 models like `davinci`
encoding_name = "cl100k_base" 

# Initialise le tokenizer avec l'encodage spécifié
tokenizer = tiktoken.get_encoding(encoding_name)

# Exemple de texte à tokeniser
text = "Bonjour, comment ça va ?"

# Tokenise le texte pour obtenir les IDs des tokens
tokensIDs = tokenizer.encode(text)

# Récupére l'encodage pour décoder les valeurs des tokens
encoding = tiktoken.get_encoding(encoding_name)

# Décode chaque ID de token en sa valeur correspondante
tokensValues = [encoding.decode_single_token_bytes(token) for token in tokensIDs]

# Affichage du text initial
print("Affichage du texte initial : ", text)  # Affiche les tokens

# Affichage des IDs des tokens
print("Affichage des tokenIDs : ", tokensIDs)  # Affiche les IDs des tokens

# Affichage des valeurs de chaque token
print("Affichage des valeurs de chaque token : ", tokensValues)  # Affiche les valeurs des tokens