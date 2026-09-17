# DocuMind — Intelligent PDF RAG Assistant

A Streamlit application based on the uploaded working RAG architecture:

**PDF → Docling → text/tables/images → Gemini visual descriptions → structure-aware chunks → Jina Embeddings v4 → FAISS → Gemini answer**

## Pages

- **Ask your PDF** — upload a PDF and chat with it.
- **About** — app explanation, usage guidance and support/contact section.

## Streamlit Secrets

Add these two secrets:

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

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Upload the project to GitHub and deploy the repository with Streamlit Community Cloud.

Before launching, add the API keys under the app's **Secrets** settings.

## Important deployment note

The original notebook was designed around Google Colab + Google Drive for persistent ingestion artifacts. This Streamlit version intentionally performs the same logical pipeline in the active app session instead of requiring the pre-generated Colab artifacts.

For very large PDFs, Docling processing, visual calls and embedding calls can be resource-intensive. A production deployment may eventually benefit from asynchronous/background processing and persistent storage.
