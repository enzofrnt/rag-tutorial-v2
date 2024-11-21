import argparse
import subprocess

from django.conf import settings
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

from .get_embedding_function import get_embedding_function

# Every answer should be in the same language as the question is asked.
# For example if the question is asked in French, the answer should be in French.
# """


def query_rag(query_text: str):
    # Préparer la base de données
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=settings.CHROMA_PATH, embedding_function=embedding_function
    )
    num_documents = len(db.get()["documents"])

    if num_documents == 0:
        # Gérer le cas où la base de données est vide
        message = "Désolé, la base de connaissances est vide. Veuillez ajouter des documents avant de poser une question."
        # Retourner un générateur qui yield le message d'erreur
        return iter([message]), []

    k = min(5, num_documents)
    results = db.similarity_search_with_score(query_text, k=k)

    # Générer le contexte à partir des documents similaires
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(settings.PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model_name = settings.LANGUAGE_MODEL_NAME
    model = OllamaLLM(model=model_name)

    # print("Prompt: ", prompt)

    # Retourner un générateur qui stream la réponse
    response_generator = model.stream(prompt)
    sources = [doc.metadata.get("id", "") for doc, _ in results]
    return response_generator, sources
