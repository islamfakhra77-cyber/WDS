from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('detect/', views.detect_image, name='detect_image'),
    path('api/detect/', views.detect_api, name='detect_api'),
    path('api/detect/frame/', views.detect_frame, name='detect_frame'),
    path('api/detect/video/', views.detect_video_api, name='detect_video'),
    path('history/', views.history, name='history'),
    path('about/', views.about, name='about'),
]