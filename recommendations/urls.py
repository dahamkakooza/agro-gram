from django.urls import path
from . import views

app_name = 'recommendations'  # This namespace exists as shown in your URLs

urlpatterns = [
    path('', views.home, name='home'),  # This is correctly defined
    path('recommend/', views.recommend, name='recommend'),
    path('download/<str:file_type>/', views.download_report, name='download_report'),
    path('chat-api/', views.chat_api, name='chat_api'),
    path('download-chat/', views.download_chat, name='download_chat'),
]