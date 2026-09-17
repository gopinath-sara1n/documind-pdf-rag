# DocuMind — Intelligent PDF Assistant

Three-file Streamlit structure:

- `app.py` — complete UI and application flow
- `rag_pipeline.py` — Docling + Jina Embeddings v4 + FAISS + Gemini RAG pipeline
- `about.py` — About & Help tab

## Streamlit Secrets

Add:

```toml
JINA_API_KEY = "your-jina-key"
GEMINI_API_KEY = "your-gemini-key"
```

Optional:

```toml
JINA_EMBEDDING_MODEL = "jina-embeddings-v4"
JINA_EMBEDDING_DIMENSIONS = "2048"
GEMINI_MODEL = "gemini-3.5-flash-lite"
```

## Deployment

Keep your existing `requirements.txt` and `packages.txt`.

The Docling pipeline explicitly uses:

```python
pipeline_options.do_ocr = False
```

to avoid the RapidOCR permission/download issue encountered during deployment.
