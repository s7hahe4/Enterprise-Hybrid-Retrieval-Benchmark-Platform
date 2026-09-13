from django.urls import path
from .views import (
    DocumentUploadView, 
    DocumentChunkListView,
    RAGQueryStreamView,
    RAGQuerySyncView,
    QueryAuditLogListView,
    BenchmarkRunExecutionView,
    BenchmarkHistoryView,
    BenchmarkDetailView,
    BenchmarkDatasetGenerateView,
    IngestionJobStatusView,
    DocumentVersionCompareView,
    DocumentRoleUpdateView
)

urlpatterns = [
    path('', DocumentUploadView.as_view(), name='document-list-api'),
    path('upload/', DocumentUploadView.as_view(), name='document-upload'),
    path('<int:pk>/chunks/', DocumentChunkListView.as_view(), name='document-chunks'),
    path('<int:pk>/role/', DocumentRoleUpdateView.as_view(), name='document-role-update'),
    # Asynchronous Ingestion Job Status
    path('jobs/<uuid:job_id>/', IngestionJobStatusView.as_view(), name='ingest-job-status'),
    # Versioning & Semantic Comparison
    path('rag/compare/', DocumentVersionCompareView.as_view(), name='document-compare'),
    # RAG Endpoints
    path('rag/stream/', RAGQueryStreamView.as_view(), name='rag-stream'),
    path('rag/query/', RAGQuerySyncView.as_view(), name='rag-query'),
    path('rag/audit/', QueryAuditLogListView.as_view(), name='rag-audit'),
    # Benchmark Evaluation Endpoints
    path('rag/benchmark/run/', BenchmarkRunExecutionView.as_view(), name='benchmark-run'),
    path('rag/benchmark/runs/', BenchmarkHistoryView.as_view(), name='benchmark-history'),
    path('rag/benchmark/runs/<int:pk>/', BenchmarkDetailView.as_view(), name='benchmark-detail'),
    path('rag/benchmark/generate-dataset/', BenchmarkDatasetGenerateView.as_view(), name='benchmark-generate-dataset'),
]