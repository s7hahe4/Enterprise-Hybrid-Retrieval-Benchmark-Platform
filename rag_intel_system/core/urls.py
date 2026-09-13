import os
from django.conf import settings
from django.views.static import serve
from django.contrib import admin
from django.urls import path, re_path, include
from django.shortcuts import render

ASSETS_DIR = os.path.join(settings.BASE_DIR, 'dist', 'assets')
if not os.path.exists(ASSETS_DIR):
    ASSETS_DIR = os.path.join(settings.BASE_DIR, 'static', 'assets')

DIST_DIR = os.path.join(settings.BASE_DIR, 'dist')

def serve_react(request):
    return render(request, 'documents/index.html')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/documents/', include('documents.urls')),
    path('documents/', include('documents.urls')),
    re_path(r'^assets/(?P<path>.*)$', serve, {'document_root': ASSETS_DIR}),
    path('favicon.svg', lambda req: serve(req, 'favicon.svg', document_root=DIST_DIR)),
    path('icons.svg', lambda req: serve(req, 'icons.svg', document_root=DIST_DIR)),
    re_path(r'^.*$', serve_react, name='react_app'),
]