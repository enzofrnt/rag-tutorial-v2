import re

from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .populate_database import add_to_chroma, load_documents, split_documents
from .query_data import query_rag


@csrf_exempt
def chat(request):
    if request.method == "POST":
        query_text = request.POST.get("query")
        response_text, sources = query_rag(query_text)

        formatted_sources = []
        for source in sources:
            match = re.search(r"rag/data/(.+):(\d+):\d+", source)
            if match:
                filename = match.group(1)
                page_number = match.group(2)
                formatted_sources.append(f"Fichier : {filename}, Page : {page_number}")
            else:
                formatted_sources.append(source)
        formatted_sources_text = "\n".join(formatted_sources)

        return JsonResponse(
            {"response": response_text, "sources": formatted_sources_text}
        )
    return render(request, "rag/chat.html")


@csrf_exempt
def add_file(request):
    if request.method == "POST" and request.FILES["file"]:
        uploaded_file = request.FILES["file"]
        file_path = default_storage.save(
            f"./rag/data/{uploaded_file.name}", uploaded_file
        )
        documents = load_documents()
        chunks = split_documents(documents)
        add_to_chroma(chunks)
        return JsonResponse({"status": "File added successfully"})
    return render(request, "rag/chat.html")
