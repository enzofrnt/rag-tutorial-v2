import os
import re

from django.conf import settings
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.utils.text import slugify
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST
from django_eventstream import send_event
from langchain_chroma import Chroma

from .get_embedding_function import get_embedding_function
from .populate_database import add_to_chroma, load_documents, split_documents, populate_database
from .query_data import query_rag

@csrf_exempt
def chat(request):
    """
    Vue permettant de gérer le système de chat basé sur un modèle RAG (Retrieval-Augmented Generation).
    Envoie les messages en temps réel via des événements serveur (Server-Sent Events).
    """
    if request.method == "POST":
        query_text = request.POST.get("query")  # Récupère la requête utilisateur
        response_generator, sources = query_rag(query_text)  # Interroge le modèle RAG
        formatted_sources_text = clean_ids(sources)  # Nettoie les identifiants des sources

        # Définir un canal d'événements pour la session
        channel_name = f"chat"

        # Envoie les réponses en morceaux via des événements serveur
        for chunk in response_generator:
            send_event(channel_name, "message", {"text": chunk})

        # Retourne les sources en réponse pour terminer
        return JsonResponse({"sources": formatted_sources_text})
    return render(request, "rag/chat.html")  # Charge la page HTML pour le chat

@csrf_exempt
def add_file(request):
    """
    Vue permettant d'ajouter des fichiers au système, de les traiter et de les insérer dans la base de données.
    """
    if request.method == "POST" and request.FILES:
        uploaded_files = request.FILES.getlist("files")  # Récupère les fichiers téléchargés
        for uploaded_file in uploaded_files:
            # Crée un nom de fichier sécurisé
            filename = slugify(os.path.splitext(uploaded_file.name)[0])
            extension = os.path.splitext(uploaded_file.name)[1]
            sanitized_filename = f"{filename}{extension}"

            # Sauvegarde le fichier dans le dossier spécifié
            os.makedirs(settings.DATA_PATH, exist_ok=True)
            file_path = os.path.join(settings.DATA_PATH, sanitized_filename)
            with open(file_path, "wb") as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)

        # Charge les documents, les segmente en morceaux et les ajoute à la base Chroma
        documents = load_documents()
        if documents:
            chunks = split_documents(documents)
            add_to_chroma(chunks)
            return JsonResponse({"status": "Files added successfully"})
        else:
            return JsonResponse({"status": "No documents to add"})

@csrf_exempt
@require_GET
def list_documents(request):
    """
    Vue permettant de lister tous les documents présents dans la base de données et de les charger s'ils ne le sont pas.
    """
    populate_database() # Charger les documents pour être sûr de tous les avoir
    
    db = Chroma(
        persist_directory=settings.CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )

    # Récupère les documents enregistrés et nettoie leurs identifiants
    documents = db.get(include=["documents"])["ids"]
    cleaned_id = clean_ids(documents)
    return JsonResponse({"documents": list(cleaned_id)})

@csrf_exempt
@require_POST
def delete_document(request):
    """
    Vue permettant de supprimer un document spécifique en fonction de son identifiant.
    """
    doc_id = request.POST.get("doc_id")  # Récupère l'ID du document à supprimer
    if not doc_id:
        return JsonResponse({"error": "ID du document manquant"}, status=400)

    delete_file_references(doc_id)  # Supprime les références associées dans la base
    delete_file(doc_id)  # Supprime le fichier du système de fichiers
    return JsonResponse({"status": "Document supprimé avec succès"})

def delete_file_references(file_name: str):
    """
    Supprime toutes les références liées à un fichier dans la base de données Chroma.
    :param file_name: Nom du fichier à rechercher dans les métadonnées (e.g., "mon_fichier.pdf").
    """
    # Charger la base de données
    db = Chroma(
        persist_directory=settings.CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )
    existing_items = db.get(include=["metadatas"])  # Récupère les métadonnées existantes
    ids_to_delete = []

    # Recherche des références associées au fichier
    for doc_id, metadata in zip(existing_items["ids"], existing_items["metadatas"]):
        source = metadata.get("source", "")
        if file_name in source:
            ids_to_delete.append(doc_id)

    # Supprime les documents identifiés
    if ids_to_delete:
        print(f"🔍 {len(ids_to_delete)} documents trouvés pour '{file_name}'. Suppression...")
        db.delete(ids=ids_to_delete)
        print("✅ Documents supprimés avec succès.")
    else:
        print(f"🚫 Aucune référence trouvée pour '{file_name}'.")

def delete_file(file_name: str):
    """
    Supprime un fichier spécifique du système de fichiers.
    :param file_name: Nom du fichier à supprimer.
    """
    file_path = os.path.join(settings.DATA_PATH, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"✅ Fichier '{file_name}' supprimé avec succès.")
    else:
        print(f"🚫 Fichier '{file_name}' introuvable.")

def clean_ids(documents):
    """
    Nettoie les identifiants des documents pour n'extraire que l'ID de base.
    :param documents: Liste des identifiants bruts.
    :return: Liste nettoyée des identifiants.
    """
    cleaned_id = set()
    for id in documents:
        cleaned_id.add(id.split(":")[0].split("/")[-1])
    return list(cleaned_id)