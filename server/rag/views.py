from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from .query_data import query_rag
from .populate_database import add_to_chroma, load_documents, split_documents

@csrf_exempt
def chat(request):
    if request.method == 'POST':
        query_text = request.POST.get('query')
        response_text = query_rag(query_text)
        return JsonResponse({'response': response_text})
    return render(request, 'rag/chat.html')

@csrf_exempt
def add_file(request):
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']
        file_path = default_storage.save(f'data/{uploaded_file.name}', uploaded_file)
        documents = load_documents()
        chunks = split_documents(documents)
        add_to_chroma(chunks)
        return JsonResponse({'status': 'File added successfully'})
    return render(request, 'rag/chat.html')
