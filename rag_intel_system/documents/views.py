import os
import re
import json
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.core.files.storage import default_storage
from django.http import StreamingHttpResponse
from django.shortcuts import render

from .models import Document, Chunk, QueryAuditLog, BenchmarkRun, IngestionJob
from .serializers import (
    DocumentSerializer, 
    ChunkSerializer, 
    QueryAuditLogSerializer, 
    BenchmarkRunSerializer,
    IngestionJobSerializer
)
from rag_engine.ingest import process_and_store_pdf, submit_async_ingest_job
from rag_engine.rag_stream import stream_rag_pipeline
from rag_engine.benchmark import run_benchmark_suite, generate_synthetic_benchmark, CONFIGURATIONS
from rag_engine.diff_engine import compare_document_versions

def index_view(request):
    """Renders the interactive web dashboard for document management and uploads."""
    return render(request, 'documents/index.html')

class DocumentUploadView(APIView):
    """
    Handles PDF document upload and listing.
    GET: Returns list of all uploaded documents.
    POST: Processes uploaded PDF asynchronously in background thread, returns job_id (HTTP 202).
    """
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        documents = Document.objects.all().order_by('-upload_date')
        serializer = DocumentSerializer(documents, many=True)
        return Response(serializer.data)

    def post(self, request, *args, **kwargs):
        file_obj = request.FILES.get('file')
        if not file_obj:
            return Response({"error": "No file provided"}, status=400)
        
        access_role = request.data.get('access_role', 'PUBLIC').upper()
        version_group = request.data.get('version_group')
        version_number = int(request.data.get('version_number', 1))
        sync_mode = request.data.get('sync', 'false').lower() in ('true', '1')

        # Auto-detect version if version_group not explicitly specified
        if not version_group:
            base_name = re.sub(r'(_v\d+|\.pdf)$', '', file_obj.name, flags=re.IGNORECASE)
            existing_count = Document.objects.filter(filename__icontains=base_name).count()
            if existing_count > 0:
                version_group = base_name
                version_number = existing_count + 1

        file_name = default_storage.save(file_obj.name, file_obj)
        file_path = default_storage.path(file_name)

        if sync_mode:
            try:
                doc = process_and_store_pdf(file_path, file_obj.name, version_group=version_group, version_number=version_number, access_role=access_role)
                if os.path.exists(file_path):
                    os.remove(file_path)
                serializer = DocumentSerializer(doc)
                return Response(serializer.data, status=201)
            except Exception as e:
                if os.path.exists(file_path):
                    os.remove(file_path)
                return Response({"error": str(e)}, status=500)

        # Async background ingestion
        job = IngestionJob.objects.create(
            filename=file_obj.name,
            status='pending',
            progress_pct=5,
            current_step='Document received and queued for ingestion'
        )

        submit_async_ingest_job(
            file_path=file_path,
            filename=file_obj.name,
            job_id=job.id,
            version_group=version_group,
            version_number=version_number,
            access_role=access_role
        )

        return Response({
            "job_id": str(job.id),
            "filename": file_obj.name,
            "status": "pending",
            "progress_pct": 5,
            "current_step": "Document received and queued for ingestion",
            "version_group": version_group,
            "version_number": version_number,
            "access_role": access_role
        }, status=202)


class DocumentChunkListView(APIView):
    """Retrieves all text chunks and metadata for a specific document."""
    def get(self, request, pk, *args, **kwargs):
        try:
            doc = Document.objects.get(pk=pk)
            chunks = doc.chunks.all().order_by('chunk_index')
            serializer = ChunkSerializer(chunks, many=True)
            return Response({
                "document_id": doc.id,
                "filename": doc.filename,
                "chunks_count": chunks.count(),
                "chunks": serializer.data
            })
        except Document.DoesNotExist:
            return Response({"error": "Document not found"}, status=404)

class RAGQueryStreamView(APIView):
    """
    Real-Time Server-Sent Events (SSE) Streaming Endpoint.
    Accepts: {"query": "..."}
    Yields: stream of SSE JSON events ('intent', 'retrieval', 'rerank', 'token', 'metrics', 'done').
    """
    def post(self, request, *args, **kwargs):
        query = request.data.get('query', '').strip()
        conversation_history = request.data.get('conversation_history', [])
        user_role = request.data.get('user_role', 'PUBLIC').upper()
        if not query:
            return Response({"error": "Query parameter is required."}, status=400)

        response = StreamingHttpResponse(
            stream_rag_pipeline(query, conversation_history=conversation_history, user_role=user_role),
            content_type='text/event-stream'
        )
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no'
        return response

class RAGQuerySyncView(APIView):
    """
    Synchronous REST endpoint for querying the RAG pipeline.
    Runs intent classification, hybrid retrieval, cross-encoder re-ranking, and response generation.
    """
    def post(self, request, *args, **kwargs):
        query = request.data.get('query', '').strip()
        conversation_history = request.data.get('conversation_history', [])
        user_role = request.data.get('user_role', 'PUBLIC').upper()
        if not query:
            return Response({"error": "Query parameter is required."}, status=400)

        events = []
        full_text = []
        metrics = {}
        intent_info = {}
        retrieval_info = {}
        rerank_info = {}
        rewrite_info = {}
        security_info = {}

        for raw_event in stream_rag_pipeline(query, conversation_history=conversation_history, user_role=user_role):

            if raw_event.startswith("data: "):
                try:
                    payload = json.loads(raw_event[6:].strip())
                    event_type = payload.get('type')
                    data = payload.get('payload', {})
                    if event_type == 'security':
                        security_info = data
                    elif event_type == 'rewrite':
                        rewrite_info = data
                    elif event_type == 'intent':
                        intent_info = data
                    elif event_type == 'retrieval':
                        retrieval_info = data
                    elif event_type == 'rerank':
                        rerank_info = data
                    elif event_type == 'token':
                        full_text.append(data.get('delta', ''))
                    elif event_type == 'metrics':
                        metrics = data
                except Exception:
                    pass

        return Response({
            'query': query,
            'standalone_query': rewrite_info.get('standalone_query', query),
            'was_rewritten': rewrite_info.get('was_rewritten', False),
            'rewrite_info': rewrite_info,
            'answer': "".join(full_text),
            'security': security_info,
            'intent': intent_info,
            'retrieval': retrieval_info,
            'rerank': rerank_info,
            'metrics': metrics
        })


class QueryAuditLogListView(APIView):
    """Returns recent MLOps audit logs with routing, re-ranking metrics, and RAGAS scores."""
    def get(self, request, *args, **kwargs):
        logs = QueryAuditLog.objects.all().order_by('-created_at')[:50]
        serializer = QueryAuditLogSerializer(logs, many=True)
        return Response(serializer.data)


class BenchmarkRunExecutionView(APIView):
    """
    Executes an empirical benchmark run across retrieval configurations.
    POST /api/documents/rag/benchmark/run/
    """
    def post(self, request, *args, **kwargs):
        name = request.data.get('name')
        num_questions = int(request.data.get('num_questions', 6))
        configurations = request.data.get('configurations', CONFIGURATIONS)
        custom_test_cases = request.data.get('custom_test_cases')

        results = run_benchmark_suite(
            test_cases=custom_test_cases,
            configurations=configurations,
            num_synthetic_questions=num_questions,
            save_run=True,
            run_name=name
        )
        return Response(results)


class BenchmarkHistoryView(APIView):
    """
    Returns historical benchmark runs.
    GET /api/documents/rag/benchmark/runs/
    """
    def get(self, request, *args, **kwargs):
        runs = BenchmarkRun.objects.all().order_by('-created_at')[:20]
        serializer = BenchmarkRunSerializer(runs, many=True)
        return Response(serializer.data)


class BenchmarkDetailView(APIView):
    """
    Retrieves specific benchmark run with detailed per-query rankings.
    GET /api/documents/rag/benchmark/runs/<id>/
    """
    def get(self, request, pk, *args, **kwargs):
        try:
            run = BenchmarkRun.objects.get(pk=pk)
            serializer = BenchmarkRunSerializer(run)
            return Response(serializer.data)
        except BenchmarkRun.DoesNotExist:
            return Response({'error': 'Benchmark run not found'}, status=404)


class BenchmarkDatasetGenerateView(APIView):
    """
    Synthesizes and returns a preview benchmark dataset from ingested documents.
    POST /api/documents/rag/benchmark/generate-dataset/
    """
    def post(self, request, *args, **kwargs):
        num_questions = int(request.data.get('num_questions', 6))
        dataset = generate_synthetic_benchmark(num_questions=num_questions)
        return Response({'count': len(dataset), 'test_cases': dataset})


class IngestionJobStatusView(APIView):
    """
    Tracks real-time progress (0-100%) and step status messages of a background PDF ingestion job.
    GET /api/documents/jobs/<uuid:job_id>/
    """
    def get(self, request, job_id, *args, **kwargs):
        try:
            job = IngestionJob.objects.get(pk=job_id)
            serializer = IngestionJobSerializer(job)
            return Response(serializer.data)
        except IngestionJob.DoesNotExist:
            return Response({'error': 'Job not found'}, status=404)


class DocumentVersionCompareView(APIView):
    """
    Compares two document versions using semantic diffing.
    POST /api/documents/rag/compare/
    Payload: {"doc_id_v1": 1, "doc_id_v2": 2, "topic_query": "..."}
    """
    def post(self, request, *args, **kwargs):
        doc_id_v1 = request.data.get('doc_id_v1')
        doc_id_v2 = request.data.get('doc_id_v2')
        topic_query = request.data.get('topic_query', '').strip()

        if not doc_id_v1 or not doc_id_v2:
            return Response({'error': 'Both doc_id_v1 and doc_id_v2 parameters are required.'}, status=400)

        result = compare_document_versions(doc_id_v1, doc_id_v2, topic_query=topic_query)
        if 'error' in result:
            return Response(result, status=400)
        return Response(result)


class DocumentRoleUpdateView(APIView):
    """
    Updates the access control role for a document.
    PATCH /api/documents/<int:pk>/role/
    Payload: {"access_role": "HR" | "ENGINEERING" | "FINANCE" | "PUBLIC" | "ADMIN"}
    """
    def patch(self, request, pk, *args, **kwargs):
        try:
            doc = Document.objects.get(pk=pk)
        except Document.DoesNotExist:
            return Response({"error": "Document not found"}, status=404)

        new_role = request.data.get('access_role', '').strip().upper()
        valid_roles = [r[0] for r in Document.ACCESS_ROLES]
        if new_role not in valid_roles:
            return Response({"error": f"Invalid role '{new_role}'. Must be one of: {valid_roles}"}, status=400)

        doc.access_role = new_role
        doc.save(update_fields=['access_role'])
        return Response({
            "id": doc.id,
            "filename": doc.filename,
            "access_role": doc.access_role,
            "message": f"Access role updated to {doc.access_role}"
        })