import os
import re

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.utils.text import slugify
from django.views.decorators.csrf import csrf_exempt

from .populate_database import add_to_chroma, load_documents, split_documents
from .query_data import query_rag


@csrf_exempt
def chat(request):
    if request.method == "POST":
        query_text = request.POST.get("query")
        response_text, sources = query_rag(query_text)

        formatted_sources_text = clean_ids(sources)

        return JsonResponse(
            {"response": response_text, "sources": formatted_sources_text}
        )
    return render(request, "rag/chat.html")


@csrf_exempt
def add_file(request):
    if request.method == "POST" and request.FILES["file"]:
        uploaded_file = request.FILES["file"]

        filename = slugify(os.path.splitext(uploaded_file.name)[0])
        extension = os.path.splitext(uploaded_file.name)[1]
        sanitized_filename = f"{filename}{extension}"

        os.makedirs(settings.DATA_PATH, exist_ok=True)
        file_path = os.path.join(settings.DATA_PATH, sanitized_filename)
        with open(file_path, "wb") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        documents = load_documents()
        chunks = split_documents(documents)
        add_to_chroma(chunks)
        return JsonResponse({"status": "File added successfully"})
    return render(request, "rag/chat.html")


from django.http import JsonResponse
from django.views.decorators.http import require_GET, require_POST
from langchain_chroma import Chroma

from .get_embedding_function import get_embedding_function


@csrf_exempt
@require_GET
def list_documents(request):
    db = Chroma(
        persist_directory=settings.CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )

    documents = db.get(include=["documents"])["ids"]
    cleaned_id = clean_ids(documents)
    return JsonResponse({"documents": list(cleaned_id)})


@csrf_exempt
@require_POST
def delete_document(request):
    doc_id = request.POST.get("doc_id")
    if not doc_id:
        return JsonResponse({"error": "ID du document manquant"}, status=400)

    delete_file_references(doc_id)

    delete_file(doc_id)

    # db.delete(ids=[doc_id])
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

    # Récupérer les documents existants
    existing_items = db.get(include=["metadatas"])  # Inclut les métadonnées
    ids_to_delete = []

    # Parcourir les métadonnées pour identifier les fichiers correspondants
    for doc_id, metadata in zip(existing_items["ids"], existing_items["metadatas"]):
        source = metadata.get("source", "")
        if file_name in source:
            ids_to_delete.append(doc_id)

    # Supprimer les documents correspondants
    if ids_to_delete:
        print(
            f"🔍 {len(ids_to_delete)} documents trouvés pour '{file_name}'. Suppression..."
        )
        db.delete(ids=ids_to_delete)
        print("✅ Documents supprimés avec succès.")
    else:
        print(f"🚫 Aucune référence trouvée pour '{file_name}'.")


def delete_file(file_name: str):
    """
    Supprime un fichier du système de fichiers.
    :param file_name: Nom du fichier à supprimer.
    """
    file_path = os.path.join(settings.DATA_PATH, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"✅ Fichier '{file_name}' supprimé avec succès.")
    else:
        print(f"🚫 Fichier '{file_name}' introuvable.")


def clean_ids(documents):
    cleaned_id = set()
    for id in documents:
        cleaned_id.add(id.split(":")[0].split("/")[-1])
    return list(cleaned_id)
