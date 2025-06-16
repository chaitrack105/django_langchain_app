from django.urls import path
from . import views

app_name = 'chats'

urlpatterns = [
    path('', views.chat_view, name='chat'),
    path('new/', views.chat_view, name='new_chat'),
]