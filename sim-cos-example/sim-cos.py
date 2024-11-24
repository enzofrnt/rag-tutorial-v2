import ollama
import numpy as np
from numpy import dot
from numpy.linalg import norm


def cosine_similarity(vec1, vec2):
    """
    Calcule la similarité cosinus entre deux vecteurs.

    La similarité cosinus mesure la similarité entre deux vecteurs
    en calculant le cosinus de l'angle entre eux.

    :param vec1: Premier vecteur (numpy array).
    :param vec2: Deuxième vecteur (numpy array).
    :return: Similarité cosinus (valeur entre -1 et 1).
    """
    # Le produit scalaire des deux vecteurs
    numerator = dot(vec1, vec2)

    # Le produit des normes (longueurs) des deux vecteurs
    denominator = norm(vec1) * norm(vec2)

    # Retourne la similarité cosinus
    return numerator / denominator


def get_vector(text):
    """
    Génère un vecteur d'embedding à partir d'un texte donné.

    Utilise le modèle 'nomic-embed-text' d'Ollama pour transformer
    un texte en une représentation vectorielle.

    :param text: Texte à convertir en vecteur.
    :return: Représentation vectorielle du texte (numpy array).
    """
    response = ollama.embeddings(
        model='nomic-embed-text',  # Modèle utilisé pour les embeddings
        prompt=text,               # Texte à convertir
    )
    return np.array(response.embedding)  # Convertit l'embedding en tableau numpy


# Comparaison de textes pour calculer leur similarité
text1 = "J'adore manger du fromage à raclette."
text2 = "J'aime manger du fromage à raclette."

vector1 = get_vector(text1)
vector2 = get_vector(text2)

similarity = cosine_similarity(vector1, vector2)
print(f"Similarité entre deux textes similaires : {similarity}")

# Exemple 2 : Textes avec peu de rapport
text1 = "Gateau"
text2 = "Voiture"

vector1 = get_vector(text1)
vector2 = get_vector(text2)

similarity = cosine_similarity(vector1, vector2)
print(f"Similarité entre deux textes sans rapport : {similarity}")

# Exemple 3 : Comparaison entre un prénom et un plat
text1 = "Emmy"
text2 = "Tarte à la menthe"

vector1 = get_vector(text1)
vector2 = get_vector(text2)

similarity = cosine_similarity(vector1, vector2)
print(f"Similarité entre un prénom et un plat : {similarity}")

# Exemple 4 : Texte numérique versus texte descriptif
text1 = "1"
text2 = "Tarte à la menthe"

vector1 = get_vector(text1)
vector2 = get_vector(text2)

similarity = cosine_similarity(vector1, vector2)
print(f"Similarité pour du texte incohérent : {similarity}")



text1 = "J'aime manger des tarte à la brique"
text2 = "000000000adfzdvaervvarvarv546455645Gdfvbfdv)p$^p`:z"

# Exemple 5 : Texte numérique versus texte descriptif
vector1 = get_vector(text1)
vector2 = get_vector(text2)

similarity = cosine_similarity(vector1, vector2)
print(f"Similarité pour du texte incohérent : {similarity}")