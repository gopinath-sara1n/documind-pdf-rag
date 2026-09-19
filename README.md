# 📘 DocuMind — Intelligent Document Assistant

DocuMind is a Streamlit-deployable PDF RAG application based on the uploaded
`final_RAG_jina_embedding2.ipynb` architecture.

## Architecture

```text
PDF
 │
 ▼
Docling
 ├── Text
 ├── Tables
 └── Pictures
       │
       ▼
     Gemini
  visual descriptions
       │
       ▼
Canonical document
       │
       ▼
Structure-aware chunks
       │
       ▼
Jina Embeddings v4
       │
       ▼
FAISS
       │
       ├── Exact word search
       │
       └── Semantic retrieval (Top 5)
                  │
                  ▼
                Gemini
                  │
                  ▼
            Grounded answer
```

## Project structure

```text
DocuMind/
├── app.py
├── rag_pipeline.py
├── about.py
├── requirements.txt
├── packages.txt
├── README.md
└── .streamlit/
    └── config.toml
```

## API keys

The app requires:

- `JINA_API_KEY`
- `GEMINI_API_KEY`

### Streamlit Community Cloud

Open your deployed app's **Settings → Secrets** and add:

```toml
JINA_API_KEY = "your-jina-key"
GEMINI_API_KEY = "your-gemini-key"
```

Do not commit these keys to GitHub.

## Run locally

Use the same Python version that you intend to use for deployment.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this project to GitHub.
2. Create a Streamlit Community Cloud app.
3. Select `app.py` as the entrypoint.
4. Add the two API keys to Streamlit Secrets.
5. Deploy.

`requirements.txt` contains Python dependencies and `packages.txt` contains
Linux dependencies used by the deployment environment.

## UI features

### Document Q&A
- PDF upload
- One-click processing
- Live progress bar and stage messages
- Pages / chunks / tables / pictures / text-element metrics
- Exact word search across chunks
- Scrollable native PDF viewer
- Chatbot-style session history
- Top-5 semantic retrieval
- Grounded Gemini answer generation
- Page / table / figure source indicators
- Expandable retrieved chunk content
- Clear document control

### About
- Purpose
- Architecture
- Data flow
- How to use
- Technology stack
- Support/contact placeholder

## Important deployment note

PDF processing is computationally heavier than ordinary Streamlit apps.
Docling, image extraction, Gemini vision calls, Jina embeddings, and FAISS
all run during document processing. Large PDFs can therefore take significant
time and may consume cloud CPU/RAM/API quotas.

For production use, consider adding:
- persistent object storage,
- persistent vector storage,
- background jobs,
- authentication,
- per-user document isolation,
- API quota monitoring,
- document-size limits.

## Source architecture fidelity

The core RAG sequence follows the uploaded notebook:

1. Docling PDF conversion.
2. Text/table/picture extraction.
3. Gemini picture descriptions.
4. Canonical document creation.
5. Structure-aware chunking with target/max character limits.
6. Jina `jina-embeddings-v4` document embeddings.
7. Normalized vectors in FAISS inner-product search.
8. Jina retrieval-query embedding.
9. Top-5 chunk retrieval.
10. Gemini answer generation using retrieved context only.

The notebook's Google Colab / Google Drive storage and `google.colab.userdata`
calls were replaced with Streamlit-compatible temporary storage and
`st.secrets`/environment variables.
