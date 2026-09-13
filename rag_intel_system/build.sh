#!/usr/bin/env bash
# exit on error
set -o errexit

pip install --upgrade pip
pip install -r requirements.txt

python manage.py migrate

# Pre-cache SentenceTransformer and CrossEncoder models during build phase
# Render provides ample RAM & CPU during build, so models are cached on disk
# and never cause network delay, thread contention, or OOM during web execution.
python -c "import os; os.environ['OMP_NUM_THREADS']='1'; import torch; torch.set_num_threads(1); from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" || true

