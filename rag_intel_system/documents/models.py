import uuid
from django.db import models

ACCESS_ROLES = [
    ('PUBLIC', 'Public (All Users)'),
    ('ENGINEERING', 'Engineering'),
    ('HR', 'Human Resources'),
    ('FINANCE', 'Finance'),
    ('ADMIN', 'Administrator (Unrestricted)'),
]

class Document(models.Model):
    ACCESS_ROLES = ACCESS_ROLES

    filename = models.CharField(max_length=255)
    upload_date = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=50, default='uploaded')
    version_group = models.CharField(max_length=100, blank=True, default='')
    version_number = models.IntegerField(default=1)
    access_role = models.CharField(max_length=50, default='PUBLIC', choices=ACCESS_ROLES)

    def __str__(self):
        v_str = f" (v{self.version_number})" if self.version_group else ""
        return f"{self.filename}{v_str} [{self.access_role}]"



class Chunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='chunks')
    parent_chunk = models.ForeignKey(
        'self', 
        null=True, 
        blank=True, 
        on_delete=models.CASCADE, 
        related_name='child_chunks'
    )
    heading = models.CharField(max_length=255, blank=True, default='')
    page_number = models.IntegerField(default=1)
    text = models.TextField()
    chunk_index = models.IntegerField()
    is_table = models.BooleanField(default=False)
    metadata = models.JSONField(default=dict, blank=True)
    
    # Note: Since we are using FAISS, we map the FAISS ID directly to this Chunk's database ID.
    def __str__(self):
        section_tag = f" [{self.heading}]" if self.heading else ""
        table_tag = " (Table)" if self.is_table else ""
        parent_tag = " (Child)" if self.parent_chunk_id else " (Parent)"
        return f"{self.document.filename} - Chunk {self.chunk_index}{section_tag}{table_tag}{parent_tag}"

class QueryLog(models.Model):
    question = models.TextField()
    answer = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.question

class QueryAuditLog(models.Model):
    """
    MLOps & Explainability audit log capturing full agentic decisions,
    query rewriting, hybrid scores, cross-encoder re-ranking metrics, latencies, and RAGAS evaluation.
    """
    original_query = models.TextField(blank=True, default='')
    question = models.TextField()  # the executed (possibly rewritten) query
    answer = models.TextField(blank=True, null=True)
    intent_classified = models.CharField(max_length=50, default='IN_DOMAIN_RAG')
    confidence = models.FloatField(default=1.0)
    was_rewritten = models.BooleanField(default=False)
    rewrite_reason = models.CharField(max_length=255, blank=True, default='')
    retrieved_chunks_count = models.IntegerField(default=0)
    user_role = models.CharField(max_length=50, default='PUBLIC')
    blocked_chunks_count = models.IntegerField(default=0)
    retrieval_metadata = models.JSONField(default=dict, blank=True)
    latency_breakdown_ms = models.JSONField(default=dict, blank=True)
    ragas_faithfulness = models.FloatField(null=True, blank=True)
    ragas_relevancy = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


    def __str__(self):
        return f"[{self.intent_classified}] {self.question[:50]} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"


class BenchmarkRun(models.Model):
    """
    Persists empirical evaluation benchmark experiments comparing retrieval configurations
    (FAISS, BM25, Naive Hybrid, Hybrid+RRF, Hybrid+RRF+Cross-Encoder).
    """
    name = models.CharField(max_length=255)
    total_queries = models.IntegerField(default=0)
    configurations = models.JSONField(default=list, blank=True)
    summary_metrics = models.JSONField(default=dict, blank=True)
    detailed_results = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} - {self.total_queries} queries ({self.created_at.strftime('%Y-%m-%d %H:%M')})"


class IngestionJob(models.Model):
    """
    Tracks non-blocking background PDF ingestion jobs with live percentage progress (0-100%)
    and step descriptions.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, null=True, blank=True, related_name='ingestion_jobs')
    filename = models.CharField(max_length=255)
    status = models.CharField(max_length=50, default='pending') # pending, extracting, tables, embedding, completed, failed
    progress_pct = models.IntegerField(default=0)
    current_step = models.CharField(max_length=255, default='Job queued')
    error_message = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Job {self.id} - {self.filename} ({self.progress_pct}% {self.status})"