from rest_framework import serializers
from .models import Document, Chunk, QueryAuditLog, BenchmarkRun, IngestionJob

class ChunkSerializer(serializers.ModelSerializer):
    parent_chunk_id = serializers.IntegerField(source='parent_chunk.id', read_only=True)

    class Meta:
        model = Chunk
        fields = [
            'id', 'chunk_index', 'heading', 'page_number', 
            'is_table', 'parent_chunk_id', 'metadata', 'text'
        ]

class DocumentSerializer(serializers.ModelSerializer):
    chunks_count = serializers.IntegerField(source='chunks.count', read_only=True)

    class Meta:
        model = Document
        fields = [
            'id', 'filename', 'upload_date', 'status', 'access_role',
            'version_group', 'version_number', 'chunks_count'
        ]

class IngestionJobSerializer(serializers.ModelSerializer):
    document_id = serializers.IntegerField(source='document.id', read_only=True)

    class Meta:
        model = IngestionJob
        fields = [
            'id', 'document_id', 'filename', 'status',
            'progress_pct', 'current_step', 'error_message',
            'created_at', 'updated_at'
        ]


class QueryAuditLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = QueryAuditLog
        fields = [
            'id', 'original_query', 'question', 'answer', 'intent_classified', 
            'confidence', 'was_rewritten', 'rewrite_reason',
            'user_role', 'blocked_chunks_count',
            'retrieved_chunks_count', 'retrieval_metadata', 'latency_breakdown_ms',
            'ragas_faithfulness', 'ragas_relevancy', 'created_at'
        ]

class BenchmarkRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = BenchmarkRun
        fields = [
            'id', 'name', 'total_queries', 'configurations',
            'summary_metrics', 'detailed_results', 'created_at'
        ]