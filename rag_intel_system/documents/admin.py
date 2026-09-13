from django.contrib import admin
from .models import Document, Chunk, QueryLog, QueryAuditLog

admin.site.register(Document)
admin.site.register(Chunk)
admin.site.register(QueryLog)
admin.site.register(QueryAuditLog)