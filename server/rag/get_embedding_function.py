from django.conf import settings
from langchain_ollama import OllamaEmbeddings

model_name = settings.EMBEDDING_MODEL_NAME


def get_embedding_function():
    model_name = settings.EMBEDDING_MODEL_NAME
    embeddings = OllamaEmbeddings(model=model_name)
    return embeddings
