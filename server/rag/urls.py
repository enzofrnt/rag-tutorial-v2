from django.urls import include, path
from django.views.generic import RedirectView
from django_eventstream import urls

from . import views

urlpatterns = [
    path("", RedirectView.as_view(url="chat/", permanent=True)),
    path("chat/", views.chat, name="chat"),
    path("add_file/", views.add_file, name="add_file"),
    path("list_documents/", views.list_documents, name="list_documents"),
    path("delete_document/", views.delete_document, name="delete_document"),
    path(
        "events/",
        include(urls),
        {"channels": ["chat"]},
    ),
]
