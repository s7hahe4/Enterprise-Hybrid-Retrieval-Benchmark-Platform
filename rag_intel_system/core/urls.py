from django.contrib import admin
from django.urls import path, include
from documents.views import index_view

urlpatterns = [
    path('', index_view, name='home'),
    path('admin/', admin.site.urls),
    path('api/documents/', include('documents.urls')),
    path('documents/', include('documents.urls')),
]