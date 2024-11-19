import argparse
import subprocess

from django.conf import settings
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from ollama._types import ResponseError

from .get_embedding_function import get_embedding_function

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
# Every answer should be in the same language as the question is asked.
# For example if the question is asked in French, the answer should be in French.
# """


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=settings.CHROMA_PATH, embedding_function=embedding_function
    )

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    try:
        model_name = settings.LANGUAGE_MODEL_NAME
        model = OllamaLLM(model=model_name)
        response_text = model.invoke(prompt)
    except ResponseError as e:
        if f"model '{model_name}' not found" in str(e):
            print(f"Model '{model_name}' not found. Pulling the model...")
            subprocess.run(["ollama", "pull", model_name], check=True)
            model = OllamaLLM(model=model_name)
            response_text = model.invoke(prompt)
        else:
            raise e

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text, sources


if __name__ == "__main__":
    main()
