from django.test import TestCase
from documents.models import Document, Chunk, BenchmarkRun, IngestionJob
from rag_engine.ingest import (
    detect_and_format_tables, 
    chunk_text_sliding, 
    is_heading_candidate, 
    parse_sections,
    update_job_progress
)
from rag_engine.bm25_search import bm25_search, get_or_build_bm25_index
from rag_engine.router import classify_intent
from rag_engine.reranker import rerank_chunks
from rag_engine.evaluator import evaluate_ragas_metrics
from rag_engine.rewriter import rewrite_query
from rag_engine.diff_engine import compare_document_versions
from rag_engine.benchmark import (
    compute_recall_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    generate_synthetic_benchmark,
    run_benchmark_suite,
    CONFIGURATIONS
)
from rag_engine.hybrid_retriever import hybrid_retrieve_candidates
from rag_engine.vector_store import add_chunks_to_vector_store
from rag_engine.rag_stream import stream_rag_pipeline
from rest_framework.test import APIClient


class RAGPipelineTests(TestCase):
    def setUp(self):
        self.doc = Document.objects.create(filename="test_report.pdf", status="completed")
        self.chunk1 = Chunk.objects.create(
            document=self.doc,
            heading="Technical Skills",
            page_number=1,
            text="Shahedul is an AI Engineer specializing in XGBoost, LightGBM, and Celery asynchronous task queues.",
            chunk_index=0,
            is_table=False
        )
        self.chunk2 = Chunk.objects.create(
            document=self.doc,
            heading="Tabular Data",
            page_number=2,
            text="| Quarter | Revenue | Growth |\n| --- | --- | --- |\n| Q1 | $10M | +15% |",
            chunk_index=1,
            is_table=True,
            metadata={"type": "table"}
        )

    def test_tabular_detection(self):
        sample_tabular_text = (
            "Company Financial Overview\n"
            "Metric    2025    2026\n"
            "Revenue   $12M    $18M\n"
            "Margin    45%     52%\n"
            "Standard paragraph about corporate operations."
        )
        prose, tables = detect_and_format_tables(sample_tabular_text)
        self.assertEqual(len(tables), 1)
        self.assertIn("| Metric | 2025 | 2026 |", tables[0])
        self.assertIn("Standard paragraph", prose)

    def test_heading_detection(self):
        self.assertEqual(is_heading_candidate("TECHNICAL SKILLS"), "Technical Skills")
        self.assertEqual(is_heading_candidate("## Financial Performance"), "Financial Performance")
        self.assertFalse(is_heading_candidate("This is just a normal sentence inside a paragraph."))

    def test_parent_child_hierarchy(self):
        parent = Chunk.objects.create(
            document=self.doc,
            heading="Experience",
            page_number=1,
            text="Full section text of the entire experience section...",
            chunk_index=2
        )
        child = Chunk.objects.create(
            document=self.doc,
            parent_chunk=parent,
            heading="Experience",
            page_number=1,
            text="[Experience] First child chunk...",
            chunk_index=3
        )
        self.assertEqual(child.parent_chunk.id, parent.id)
        self.assertEqual(parent.child_chunks.count(), 1)
        self.assertEqual(parent.child_chunks.first().id, child.id)

    def test_query_rewriter(self):
        # 1. First turn standalone query
        res1 = rewrite_query("What algorithms did Shahedul use?", conversation_history=[])
        self.assertFalse(res1['was_rewritten'])
        self.assertEqual(res1['standalone_query'], "What algorithms did Shahedul use?")

        # 2. Multi-turn ambiguous follow-up
        history = [
            {"role": "user", "content": "What machine learning frameworks does Shahedul use?"},
            {"role": "assistant", "content": "Shahedul uses XGBoost, Scikit-Learn, and PyTorch."}
        ]
        res2 = rewrite_query("What about his deep learning work?", conversation_history=history)
        self.assertTrue(res2['was_rewritten'])
        self.assertIn("deep learning", res2['standalone_query'].lower())

    def test_bm25_search(self):
        get_or_build_bm25_index(force_rebuild=True)
        results = bm25_search("XGBoost Celery task queues", top_k=5)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]['chunk_id'], self.chunk1.id)

    def test_intent_router_chitchat(self):
        result = classify_intent("Hello! Who are you?")
        self.assertEqual(result['intent'], 'CHITCHAT')
        self.assertGreaterEqual(result['confidence'], 0.90)

    def test_intent_router_ood(self):
        result = classify_intent("How to bake chocolate cookies in the kitchen?")
        self.assertIn(result['intent'], ['OUT_OF_DOMAIN', 'CHITCHAT'])

    def test_cross_encoder_reranker(self):
        candidates = [
            {
                'chunk_id': self.chunk1.id,
                'chunk_index': 0,
                'document_filename': self.doc.filename,
                'text': self.chunk1.text,
                'is_table': False,
                'initial_rrf_rank': 1
            },
            {
                'chunk_id': self.chunk2.id,
                'chunk_index': 1,
                'document_filename': self.doc.filename,
                'text': self.chunk2.text,
                'is_table': True,
                'initial_rrf_rank': 2
            }
        ]
        top_k, full = rerank_chunks("What ML algorithms were used?", candidates, top_k=2)
        self.assertEqual(len(top_k), 2)
        self.assertTrue(0.0 <= top_k[0]['relevance_score'] <= 1.0)
        self.assertIn('rank_delta', top_k[0])

    def test_ragas_evaluator(self):
        query = "What algorithms does Shahedul use?"
        answer = "Shahedul uses XGBoost and LightGBM for machine learning tasks."
        context = [{'text': self.chunk1.text}]
        
        metrics = evaluate_ragas_metrics(query, answer, context)
        self.assertIn('faithfulness', metrics)
        self.assertIn('relevancy', metrics)
        self.assertGreater(metrics['faithfulness'], 0.5)
        self.assertGreater(metrics['relevancy'], 0.5)

    def test_ir_metrics_calculations(self):
        retrieved = [10, 20, 30, 40, 50]
        relevant = {30, 60}

        # Recall@3: chunk 30 is at index 2 (position 3) -> 1 of 2 relevant retrieved -> 0.5
        r3 = compute_recall_at_k(retrieved, relevant, k=3)
        self.assertEqual(r3, 0.5)

        # Recall@2: chunk 30 is at position 3 -> 0 relevant retrieved in top 2 -> 0.0
        r2 = compute_recall_at_k(retrieved, relevant, k=2)
        self.assertEqual(r2, 0.0)

        # MRR: first relevant hit is at position 3 -> 1/3
        mrr = compute_mrr(retrieved, relevant)
        self.assertAlmostEqual(mrr, 1.0 / 3.0, places=4)

        # Precision@3: 1 of 3 retrieved items is relevant -> 1/3
        p3 = compute_precision_at_k(retrieved, relevant, k=3)
        self.assertAlmostEqual(p3, 1.0 / 3.0, places=4)

        # NDCG@5: calculation check
        ndcg = compute_ndcg_at_k(retrieved, relevant, k=5)
        self.assertGreater(ndcg, 0.0)
        self.assertLessEqual(ndcg, 1.0)

    def test_synthetic_benchmark_generator(self):
        test_cases = generate_synthetic_benchmark(num_questions=2)
        self.assertTrue(len(test_cases) > 0)
        tc = test_cases[0]
        self.assertIn('query', tc)
        self.assertIn('relevant_chunk_ids', tc)
        self.assertTrue(len(tc['relevant_chunk_ids']) > 0)

    def test_benchmark_suite_execution(self):
        custom_test_cases = [
            {
                'id': 1,
                'query': "What are the details regarding Technical Skills?",
                'relevant_chunk_ids': [self.chunk1.id],
                'target_heading': "Technical Skills",
                'document_filename': self.doc.filename,
                'passage_excerpt': self.chunk1.text[:50]
            }
        ]
        result = run_benchmark_suite(
            test_cases=custom_test_cases,
            configurations=CONFIGURATIONS,
            save_run=True,
            run_name="Test Benchmark"
        )
        self.assertIn('summary_metrics', result)
        self.assertEqual(len(result['summary_metrics']), len(CONFIGURATIONS))
        self.assertIn('detailed_results', result)
        self.assertEqual(len(result['detailed_results']), 1)
        self.assertIsNotNone(result.get('benchmark_run_id'))
        
        # Verify persistence
        run_obj = BenchmarkRun.objects.get(pk=result['benchmark_run_id'])
        self.assertEqual(run_obj.name, "Test Benchmark")
        self.assertEqual(run_obj.total_queries, 1)

    def test_benchmark_api_endpoints(self):
        client = APIClient()
        # 1. Trigger benchmark run via API
        resp = client.post('/api/documents/rag/benchmark/run/', {'num_questions': 2}, format='json')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('summary_metrics', resp.data)
        run_id = resp.data.get('benchmark_run_id')
        self.assertIsNotNone(run_id)

        # 2. Fetch history
        history_resp = client.get('/api/documents/rag/benchmark/runs/')
        self.assertEqual(history_resp.status_code, 200)
        self.assertTrue(len(history_resp.data) > 0)

        # 3. Fetch detail
        detail_resp = client.get(f'/api/documents/rag/benchmark/runs/{run_id}/')
        self.assertEqual(detail_resp.status_code, 200)
        self.assertEqual(detail_resp.data['id'], run_id)

    def test_async_ingestion_job_lifecycle(self):
        job = IngestionJob.objects.create(
            filename="async_test.pdf",
            status="pending",
            progress_pct=5,
            current_step="Queued"
        )
        self.assertEqual(job.progress_pct, 5)
        self.assertEqual(job.status, "pending")

        # Update progress
        update_job_progress(job.id, status="extracting", progress_pct=25, current_step="Extracting text")
        job.refresh_from_db()
        self.assertEqual(job.status, "extracting")
        self.assertEqual(job.progress_pct, 25)

        # Mark completed
        update_job_progress(job.id, status="completed", progress_pct=100, current_step="Done")
        job.refresh_from_db()
        self.assertEqual(job.status, "completed")
        self.assertEqual(job.progress_pct, 100)

    def test_document_versioning(self):
        doc_v1 = Document.objects.create(
            filename="Policy.pdf",
            version_group="Policy",
            version_number=1,
            status="completed"
        )
        doc_v2 = Document.objects.create(
            filename="Policy.pdf",
            version_group="Policy",
            version_number=2,
            status="completed"
        )
        self.assertEqual(doc_v1.version_group, doc_v2.version_group)
        self.assertEqual(doc_v1.version_number, 1)
        self.assertEqual(doc_v2.version_number, 2)

    def test_semantic_diff_engine(self):
        # Create two document versions with slight differences
        doc1 = Document.objects.create(filename="agreement_v1.pdf", version_number=1, status="completed")
        Chunk.objects.create(
            document=doc1,
            heading="Terms",
            page_number=1,
            text="Employees must work from the physical office 5 days a week. Core hours are 9 AM to 5 PM.",
            chunk_index=0
        )

        doc2 = Document.objects.create(filename="agreement_v2.pdf", version_number=2, status="completed")
        Chunk.objects.create(
            document=doc2,
            heading="Terms",
            page_number=1,
            text="Employees may work remotely 3 days a week. Core hours are 9 AM to 5 PM. A home office stipend is provided.",
            chunk_index=0
        )

        diff = compare_document_versions(doc1.id, doc2.id, topic_query="remote work")
        self.assertIn('metrics', diff)
        self.assertIn('summary_diff', diff)
        self.assertTrue(diff['metrics']['added_count'] >= 1 or diff['metrics']['modified_count'] >= 1)

    def test_document_rbac_defaults_and_api_update(self):
        doc = Document.objects.create(filename="employee_handbook.pdf", status="completed")
        self.assertEqual(doc.access_role, "PUBLIC")

        client = APIClient()
        # 1. Update role to HR
        resp = client.patch(f'/api/documents/{doc.id}/role/', {'access_role': 'HR'}, format='json')
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.data['access_role'], 'HR')
        doc.refresh_from_db()
        self.assertEqual(doc.access_role, 'HR')

        # 2. Reject invalid role
        bad_resp = client.patch(f'/api/documents/{doc.id}/role/', {'access_role': 'SUPERUSER'}, format='json')
        self.assertEqual(bad_resp.status_code, 400)

    def test_rbac_retrieval_filtering(self):
        # Create HR document with sensitive salary chunk
        hr_doc = Document.objects.create(filename="hr_salaries.pdf", access_role="HR", status="completed")
        hr_chunk = Chunk.objects.create(
            document=hr_doc,
            heading="Executive Compensation",
            page_number=1,
            text="Executive bonuses and base salary compensation breakdown for Q4.",
            chunk_index=0
        )
        get_or_build_bm25_index(force_rebuild=True)

        # 1. User with role PUBLIC should have HR chunk blocked
        candidates_public = hybrid_retrieve_candidates("Executive salary compensation", top_k=10, user_role="PUBLIC")
        chunk_ids_public = [c['chunk_id'] for c in candidates_public]
        self.assertNotIn(hr_chunk.id, chunk_ids_public)
        self.assertGreaterEqual(candidates_public.blocked_count, 1)

        # 2. User with role HR should have access
        candidates_hr = hybrid_retrieve_candidates("Executive salary compensation", top_k=10, user_role="HR")
        chunk_ids_hr = [c['chunk_id'] for c in candidates_hr]
        self.assertIn(hr_chunk.id, chunk_ids_hr)

        # 3. User with role ADMIN should have access
        candidates_admin = hybrid_retrieve_candidates("Executive salary compensation", top_k=10, user_role="ADMIN")
        chunk_ids_admin = [c['chunk_id'] for c in candidates_admin]
        self.assertIn(hr_chunk.id, chunk_ids_admin)

    def test_rbac_stream_rejection_guardrail(self):
        # Isolate database for this test
        Chunk.objects.all().delete()
        Document.objects.all().delete()

        # Query exclusively targeting an HR document with ENGINEERING role
        hr_doc = Document.objects.create(filename="private_hr_secrets.pdf", access_role="HR", status="completed")
        c = Chunk.objects.create(
            document=hr_doc,
            heading="Confidential HR",
            page_number=1,
            text="Strictly confidential payroll ledger and executive equity grants.",
            chunk_index=0
        )
        add_chunks_to_vector_store([c.id], [c.text])
        get_or_build_bm25_index(force_rebuild=True)

        events = []
        for raw in stream_rag_pipeline("confidential payroll ledger equity grants", user_role="ENGINEERING"):
            events.append(raw)
        stream_text = "".join(events)
        self.assertIn("security", stream_text)
        self.assertIn("Access", stream_text)
        self.assertIn("Restricted", stream_text)
        self.assertNotIn("payroll ledger and executive equity grants", stream_text)



