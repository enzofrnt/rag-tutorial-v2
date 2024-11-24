import argparse
import os
import shutil

from django.conf import settings
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .get_embedding_function import get_embedding_function


def populate_database():
    """
    Charge les documents, les segmente en morceaux et les ajoute à la base de données Chroma.
    """
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def reset_database():
    """
    Réinitialise complètement la base de données en supprimant tout son contenu.
    """
    clear_database()


def load_documents():
    """
    Charge les documents PDF depuis un répertoire spécifié dans les paramètres Django.
    
    :return: Liste de documents chargés.
    """
    document_loader = PyPDFDirectoryLoader(settings.DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    """
    Divise les documents en morceaux de taille contrôlée pour l'indexation.

    :param documents: Liste de documents à segmenter.
    :return: Liste de morceaux de texte segmentés.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,           # Taille maximale d'un morceau (en caractères).
        chunk_overlap=80,         # Chevauchement entre les morceaux pour la continuité.
        length_function=len,      # Fonction pour mesurer la longueur des morceaux.
        is_separator_regex=False, # Indique que le séparateur n'est pas une expression régulière.
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    """
    Ajoute des morceaux de texte à la base de données Chroma.
    Évite d'ajouter des documents déjà présents dans la base.

    :param chunks: Liste de morceaux de texte à ajouter.
    """
    # Charger la base de données existante
    db = Chroma(
        persist_directory=settings.CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )

    # Calculer les identifiants uniques pour chaque morceau
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Vérifier les documents existants dans la base
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Nombre de documents existants dans la base : {len(existing_ids)}")

    # Ajouter uniquement les nouveaux documents
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"👉 Ajout de nouveaux documents : {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ Aucun nouveau document à ajouter")


def calculate_chunk_ids(chunks):
    """
    Calcule des identifiants uniques pour chaque morceau basé sur la source, la page et l'index.

    :param chunks: Liste de morceaux de texte.
    :return: Liste de morceaux avec des identifiants ajoutés.
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")  # Source du document (par exemple, le nom du fichier PDF).
        page = chunk.metadata.get("page")     # Numéro de la page.
        current_page_id = f"{source}:{page}"  # ID unique pour la page (source:page).

        # Incrémente l'index si le morceau appartient à la même page que le précédent.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Génère un ID unique pour le morceau.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Ajoute l'ID au metadata du morceau.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    """
    Supprime complètement le contenu de la base de données Chroma en effaçant le dossier correspondant.

    :return: None
    """
    if os.path.exists(settings.CHROMA_PATH):
        shutil.rmtree(settings.CHROMA_PATH)