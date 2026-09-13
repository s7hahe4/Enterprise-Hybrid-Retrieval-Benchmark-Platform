import os
from django.conf import settings
from django.http import FileResponse, HttpResponseNotFound
from django.views.static import serve
from django.contrib import admin
from django.urls import path, re_path, include

DIST_DIR = os.path.join(settings.BASE_DIR, "dist")

def serve_react(request):
    index_path = os.path.join(DIST_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(open(index_path, "rb"), content_type="text/html")
    return HttpResponseNotFound("React build not found. Run npm run build.")

def serve_favicon(request):
    fav_path = os.path.join(DIST_DIR, "favicon.svg")
    if os.path.exists(fav_path):
        return FileResponse(open(fav_path, "rb"), content_type="image/svg+xml")
    return HttpResponseNotFound()

def serve_icons(request):
    icons_path = os.path.join(DIST_DIR, "icons.svg")
    if os.path.exists(icons_path):
        return FileResponse(open(icons_path, "rb"), content_type="image/svg+xml")
    return HttpResponseNotFound()

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/documents/', include('documents.urls')),
    path('documents/', include('documents.urls')),
    re_path(r'^assets/(?P<path>.*)$', serve, {'document_root': os.path.join(DIST_DIR, 'assets')}),
    path('favicon.svg', serve_favicon, name='favicon'),
    path('icons.svg', serve_icons, name='icons'),
    re_path(r'^.*$', serve_react, name='react_app'),
]