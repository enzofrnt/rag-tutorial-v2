from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chat, name='chat'),
    path('add_file/', views.add_file, name='add_file'),
]
