import argparse
import subprocess

from django.conf import settings
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

from .get_embedding_function import get_embedding_function


def query_rag(query_text: str):
    """
    Fonction pour interroger un système RAG (Retrieval-Augmented Generation).
    Utilise une base de connaissances (Chroma) et un modèle de langage (OllamaLLM)
    pour répondre à une question donnée en fonction de documents similaires.

    :param query_text: Texte de la requête utilisateur.
    :return: Un générateur de réponse (streaming) et une liste des sources utilisées.
    """
    # Préparer la fonction d'embedding et la base de données
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=settings.CHROMA_PATH, embedding_function=embedding_function
    )
    num_documents = len(db.get()["documents"])  # Nombre de documents dans la base

    if num_documents == 0:
        # Gérer le cas où la base de données est vide
        message = "Désolé, la base de connaissances est vide. Veuillez ajouter des documents avant de poser une question."
        # Retourner un générateur qui yield le message d'erreur
        return iter([message]), []

    # Limiter le nombre de documents retournés pour la recherche par similarité
    k = min(5, num_documents)
    results = db.similarity_search_with_score(
        query_text, k=k
    )  # Recherche des documents les plus proches

    # Générer le contexte à partir des documents similaires
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(
        settings.PROMPT_TEMPLATE
    )  # Charger le modèle de prompt
    prompt = prompt_template.format(
        context=context_text, question=query_text
    )  # Insérer le contexte et la question dans le prompt

    # Afficher le prompt
    # print(f"Prompt:\n{prompt}")

    # Charger le modèle de langage configuré
    model_name = settings.LANGUAGE_MODEL_NAME
    model = OllamaLLM(model=model_name)

    # Retourner un générateur qui stream la réponse
    response_generator = model.stream(prompt)
    sources = [
        doc.metadata.get("id", "") for doc, _ in results
    ]  # Extraire les identifiants des documents sources
    return response_generator, sources


from pgvector.django import CosineDistance

from .models import Chunk


def get_similar_chunks(query_embedding, top_k=5):
    """
    Trouve les chunks les plus similaires à un embedding donné en utilisant la distance cosinus.

    :param query_embedding: Embedding de la requête utilisateur (liste de flottants).
    :param top_k: Nombre de résultats les plus proches à retourner.
    :return: Liste des chunks et leurs distances.
    """
    # Effectuer la recherche avec CosineDistance
    similar_chunks = Chunk.objects.annotate(
        similarity=CosineDistance("embedding", query_embedding)
    ).order_by(
        "similarity"
    )[  # Distance cosinus croissante (plus proche = meilleur)
        :top_k
    ]  # Limite à top_k résultats

    return similar_chunks


from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from .get_embedding_function import get_embedding_function
from .models import Chunk


def query_rag_with_postgres(query_text: str):
    """
    Interroge une base PostgreSQL pour récupérer des chunks similaires,
    puis utilise un modèle de langage pour répondre.

    :param query_text: Question utilisateur.
    :return: Générateur de réponse et liste des sources.
    """
    # Générer l'embedding pour la requête
    embedding_function = get_embedding_function()
    query_embedding = embedding_function.embed_query(query_text)

    # Rechercher les chunks similaires
    similar_chunks = get_similar_chunks(query_embedding)

    if not similar_chunks:
        return iter(["Désolé, aucun document pertinent trouvé."]), []

    # Générer le contexte à partir des chunks
    context_text = "\n\n---\n\n".join([chunk.content for chunk in similar_chunks])

    # Charger le modèle de langage
    prompt_template = ChatPromptTemplate.from_template(settings.PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = OllamaLLM(model=settings.LANGUAGE_MODEL_NAME)

    # Streamer la réponse et collecter les sources
    response_generator = model.stream(prompt)
    sources = [
        f"{chunk.document.file.name}: Page {chunk.page}, Chunk {chunk.chunk_index}"
        for chunk in similar_chunks
    ]

    return response_generator, sources
