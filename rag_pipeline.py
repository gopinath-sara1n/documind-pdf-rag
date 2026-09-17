import os
import re
import tempfile
from pathlib import Path

import faiss
import numpy as np
import requests
import streamlit as st

from google import genai
from google.genai import types

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


# ============================================================
# Configuration
# ============================================================
JINA_MODEL = os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v4")
JINA_DIMENSIONS = int(os.getenv("JINA_EMBEDDING_DIMENSIONS", "2048"))
JINA_ENDPOINT = "https://api.jina.ai/v1/embeddings"
JINA_BATCH_SIZE = 100

GEMINI_MODELS = [
    os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
]


# ============================================================
# Secrets
# ============================================================
def get_secret(name):
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name)


def get_jina_key():
    key = get_secret("JINA_API_KEY")
    if not key:
        raise RuntimeError("JINA_API_KEY is not configured in Streamlit Secrets.")
    return key


def get_gemini_key():
    key = get_secret("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured in Streamlit Secrets.")
    return key


# ============================================================
# Gemini client
# ============================================================
@st.cache_resource(show_spinner=False)
def get_gemini_client():
    return genai.Client(api_key=get_gemini_key())


# ============================================================
# Docling
# ============================================================
@st.cache_resource(show_spinner=False)
def get_converter():
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = True
    pipeline_options.do_picture_description = False

    # Important for Streamlit deployment:
    # avoids RapidOCR model permission/download problems when OCR
    # is not required by the application.
    pipeline_options.do_ocr = False

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )


# ============================================================
# Text helpers
# ============================================================
def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_item_text(item):
    for attr in ("text", "export_to_text"):
        try:
            value = getattr(item, attr, None)
            if callable(value):
                value = value()
            if value:
                return clean_text(value)
        except Exception:
            pass
    return ""


def get_page_number(item):
    candidates = [
        getattr(item, "page_no", None),
        getattr(item, "page", None),
    ]

    for candidate in candidates:
        if isinstance(candidate, int):
            return candidate
        if candidate is not None:
            try:
                return int(candidate)
            except Exception:
                pass

    try:
        prov = getattr(item, "prov", None)
        if prov:
            first = prov[0]
            page_no = getattr(first, "page_no", None)
            if page_no is not None:
                return int(page_no)
    except Exception:
        pass

    return None


def table_to_text(table):
    try:
        return clean_text(table.export_to_markdown())
    except Exception:
        pass

    try:
        return clean_text(table.export_to_text())
    except Exception:
        pass

    return clean_text(str(table))


def split_sentences(text, target_chars=1800, max_chars=3200):
    text = clean_text(text)
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if current and len(current) + len(sentence) + 1 > target_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()

        if len(current) >= max_chars:
            chunks.append(current[:max_chars].strip())
            current = current[max_chars:].strip()

    if current:
        chunks.append(current)

    return chunks


# ============================================================
# Gemini visual descriptions
# ============================================================
def classify_gemini_error(exc):
    text = str(exc).lower()

    if "quota" in text or "429" in text:
        return "Gemini quota/rate limit was reached."
    if "503" in text or "unavailable" in text:
        return "The Gemini model is temporarily unavailable."
    if "not found" in text or "404" in text:
        return "The configured Gemini model is unavailable for this API."
    return str(exc)


def describe_visual(image):
    client = get_gemini_client()

    prompt = """
Describe this PDF visual for document retrieval.

Include:
- what the visual represents
- title/caption if visible
- important labels
- important values or trends
- relationships or comparisons
- any useful context

Do not invent information that is not visible.
Return a concise factual description.
"""

    last_error = None

    for model in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(
                        data=image,
                        mime_type="image/png",
                    ),
                    prompt,
                ],
            )
            text = getattr(response, "text", None)
            if text:
                return clean_text(text)
        except Exception as exc:
            last_error = exc

    if last_error:
        raise RuntimeError(classify_gemini_error(last_error))

    return ""


# ============================================================
# Structure-aware chunking
# ============================================================
def make_chunks(elements, target_chars=1800, max_chars=3200):
    chunks = []

    for element in elements:
        kind = element.get("type", "text")
        page = element.get("page")
        content = clean_text(element.get("content", ""))

        if not content:
            continue

        if kind in ("table", "visual"):
            chunks.append({
                "text": content,
                "type": kind,
                "page": page,
            })
            continue

        parts = split_sentences(
            content,
            target_chars=target_chars,
            max_chars=max_chars,
        )

        for part in parts:
            chunks.append({
                "text": part,
                "type": "text",
                "page": page,
            })

    return chunks


# ============================================================
# Jina Embeddings
# ============================================================
def jina_request(texts, task):
    headers = {
        "Authorization": f"Bearer {get_jina_key()}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": JINA_MODEL,
        "input": texts,
        "task": task,
        "dimensions": JINA_DIMENSIONS,
    }

    response = requests.post(
        JINA_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=180,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Jina Embeddings API error {response.status_code}: "
            f"{response.text[:500]}"
        )

    data = response.json()["data"]
    data = sorted(data, key=lambda x: x["index"])

    return [item["embedding"] for item in data]


def format_document_for_embedding(chunk):
    kind = chunk.get("type", "text")
    page = chunk.get("page")

    prefix = f"[Page {page}] " if page is not None else ""

    if kind == "table":
        prefix += "[TABLE] "
    elif kind == "visual":
        prefix += "[VISUAL] "

    return prefix + chunk["text"]


def embed_documents(chunks):
    all_embeddings = []

    for start in range(0, len(chunks), JINA_BATCH_SIZE):
        batch = chunks[start:start + JINA_BATCH_SIZE]
        texts = [format_document_for_embedding(x) for x in batch]

        embeddings = jina_request(texts, task="retrieval.passage")
        all_embeddings.extend(embeddings)

    return np.asarray(all_embeddings, dtype="float32")


def embed_query(query):
    embeddings = jina_request(
        [query],
        task="retrieval.query",
    )
    return np.asarray(embeddings[0], dtype="float32")


# ============================================================
# PDF extraction
# ============================================================
def extract_pdf(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        pdf_path = tmp.name

    converter = get_converter()
    result = converter.convert(pdf_path)
    doc = result.document

    elements = []
    table_count = 0
    visual_count = 0

    # Main document items
    try:
        for item, level in doc.iterate_items():
            item_type = item.__class__.__name__.lower()

            page = get_page_number(item)

            if "table" in item_type:
                text = table_to_text(item)
                if text:
                    elements.append({
                        "type": "table",
                        "page": page,
                        "content": text,
                    })
                    table_count += 1
                continue

            if "picture" in item_type:
                # Pictures are handled separately below.
                continue

            text = get_item_text(item)
            if text:
                elements.append({
                    "type": "text",
                    "page": page,
                    "content": text,
                })
    except Exception:
        # Fallback for Docling versions with different iteration behavior.
        pass

    # Pictures / visuals
    try:
        pictures = getattr(doc, "pictures", [])
        for picture in pictures:
            page = get_page_number(picture)

            image = None

            try:
                image_ref = picture.get_image(doc)
                if image_ref is not None:
                    from io import BytesIO
                    buffer = BytesIO()
                    image_ref.save(buffer, format="PNG")
                    image = buffer.getvalue()
            except Exception:
                image = None

            if image:
                try:
                    description = describe_visual(image)
                except Exception as exc:
                    description = f"Visual description unavailable: {exc}"

                if description:
                    elements.append({
                        "type": "visual",
                        "page": page,
                        "content": description,
                    })
                    visual_count += 1
    except Exception:
        pass

    elements.sort(
        key=lambda x: (
            x.get("page") if x.get("page") is not None else 10**9
        )
    )

    chunks = make_chunks(elements)
    embeddings = embed_documents(chunks)

    # Cosine similarity through normalized inner product.
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    pages = 0

    try:
        pages = int(getattr(doc, "num_pages", 0) or 0)
    except Exception:
        pass

    if not pages:
        pages = max(
            [x["page"] for x in elements if x.get("page") is not None] or [0]
        )

    metadata = [
        {
            "page": chunk.get("page"),
            "type": chunk.get("type", "text"),
        }
        for chunk in chunks
    ]

    stats = {
        "tables": table_count,
        "visuals": visual_count,
    }

    try:
        os.remove(pdf_path)
    except Exception:
        pass

    return {
        "chunks": chunks,
        "metadata": metadata,
        "index": index,
        "pages": pages,
        "stats": stats,
    }


# ============================================================
# Public processing function
# ============================================================
def process_pdf(uploaded_file):
    return extract_pdf(uploaded_file)


# ============================================================
# Retrieval
# ============================================================
def retrieve(question, index, chunks, metadata, top_k=5):
    if index is None or not chunks:
        return []

    query_vector = embed_query(question).reshape(1, -1)
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, min(top_k, len(chunks)))

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        results.append({
            "score": float(score),
            "chunk": chunks[idx],
            "metadata": metadata[idx],
        })

    return results


def build_context(results):
    blocks = []

    for i, result in enumerate(results, start=1):
        chunk = result["chunk"]
        metadata = result["metadata"]

        page = metadata.get("page")
        kind = metadata.get("type", "text")

        page_text = f"Page {page}" if page is not None else "Page unknown"

        blocks.append(
            f"[Retrieved Source {i} | {page_text} | {kind}]\n"
            f"{chunk.get('text', '')}"
        )

    return "\n\n".join(blocks)


# ============================================================
# Gemini answer generation
# ============================================================
def generate_answer(question, results):
    if not results:
        return "I couldn't find relevant information in the document."

    context = build_context(results)
    client = get_gemini_client()

    prompt = f"""
You are DocuMind, an intelligent PDF assistant.

Answer the user's question using ONLY the retrieved document context below.

Rules:
1. Do not invent facts.
2. Do not use outside knowledge.
3. If the context does not contain enough information, say that the
   document context does not provide enough information.
4. When useful, mention page numbers.
5. Be clear and concise.
6. For tables or charts, explain the relevant values or trends from
   the retrieved context.

RETRIEVED DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    last_error = None

    for model in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )

            text = getattr(response, "text", None)
            if text:
                return text.strip()

        except Exception as exc:
            last_error = exc

    if last_error:
        raise RuntimeError(classify_gemini_error(last_error))

    return "I could not generate an answer from the retrieved document context."


# ============================================================
# Cache cleanup
# ============================================================
def clear_backend_cache():
    # The actual document state lives in Streamlit session_state.
    # Cached resources are intentionally kept because rebuilding the
    # Docling/Gemini clients is expensive.
    return None
