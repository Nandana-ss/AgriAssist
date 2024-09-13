
from django.contrib import admin
from django.urls import include, path
from AgriApp import views
from AgriApp.views import disease_recognition 
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),       
    path('', include('AgriApp.urls')),     
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)