import os
import re
import tempfile
from pathlib import Path
from io import BytesIO

import faiss
import numpy as np
import pandas as pd
import requests
import streamlit as st

from google import genai
from google.genai import types

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


# ============================================================
# CONFIGURATION
# ============================================================

JINA_MODEL = os.getenv(
    "JINA_EMBEDDING_MODEL",
    "jina-embeddings-v4"
)

JINA_DIMENSIONS = int(
    os.getenv(
        "JINA_EMBEDDING_DIMENSIONS",
        "2048"
    )
)

JINA_ENDPOINT = "https://api.jina.ai/v1/embeddings"

JINA_BATCH_SIZE = 100

GEMINI_MODELS = [
    os.getenv(
        "GEMINI_MODEL",
        "gemini-3.5-flash-lite"
    ),
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
]


# ============================================================
# API KEYS
# ============================================================

def get_secret(name):
    """
    Get a secret from Streamlit Secrets first,
    then fall back to environment variables.
    """

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
        raise RuntimeError(
            "JINA_API_KEY is not configured. "
            "Add it to Streamlit Secrets."
        )

    return key


def get_gemini_key():

    key = get_secret("GEMINI_API_KEY")

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. "
            "Add it to Streamlit Secrets."
        )

    return key


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource(show_spinner=False)
def get_gemini_client():

    return genai.Client(
        api_key=get_gemini_key()
    )


# ============================================================
# DOCLING CONVERTER
# ============================================================

@st.cache_resource(show_spinner=False)
def get_converter():

    pipeline_options = PdfPipelineOptions()

    # Table extraction
    pipeline_options.do_table_structure = True

    # Generate picture images
    pipeline_options.generate_picture_images = True

    # We handle visual descriptions ourselves
    pipeline_options.do_picture_description = False

    # IMPORTANT:
    # Prevent RapidOCR model permission/download issue
    # seen on Streamlit deployment.
    pipeline_options.do_ocr = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    return converter


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def get_item_text(item):

    # Try text property
    try:

        value = getattr(
            item,
            "text",
            None
        )

        if value:

            return clean_text(value)

    except Exception:
        pass

    # Try export_to_text()
    try:

        method = getattr(
            item,
            "export_to_text",
            None
        )

        if callable(method):

            value = method()

            if value:

                return clean_text(value)

    except Exception:
        pass

    return ""


def get_page_number(item):

    # Direct page_no
    try:

        page_no = getattr(
            item,
            "page_no",
            None
        )

        if page_no is not None:

            return int(page_no)

    except Exception:
        pass

    # Direct page
    try:

        page = getattr(
            item,
            "page",
            None
        )

        if page is not None:

            return int(page)

    except Exception:
        pass

    # Provenance
    try:

        prov = getattr(
            item,
            "prov",
            None
        )

        if prov:

            first_prov = prov[0]

            page_no = getattr(
                first_prov,
                "page_no",
                None
            )

            if page_no is not None:

                return int(page_no)

    except Exception:
        pass

    return None


# ============================================================
# TABLE EXTRACTION
# ============================================================

def table_to_text(table, doc):

    """
    Convert Docling TableItem to searchable text.

    IMPORTANT:
    Current Docling versions require doc=doc.
    """

    # --------------------------------------------------------
    # Preferred method: DataFrame
    # --------------------------------------------------------

    try:

        df = table.export_to_dataframe(
            doc=doc
        )

        if df is not None:

            if isinstance(df, pd.DataFrame):

                if not df.empty:

                    return clean_text(
                        df.to_markdown(
                            index=False
                        )
                    )

    except Exception:
        pass

    # --------------------------------------------------------
    # Fallback: Markdown
    # --------------------------------------------------------

    try:

        markdown = table.export_to_markdown(
            doc=doc
        )

        if markdown:

            return clean_text(markdown)

    except Exception:
        pass

    # --------------------------------------------------------
    # Final fallback
    # --------------------------------------------------------

    try:

        return clean_text(
            str(table)
        )

    except Exception:

        return ""


# ============================================================
# SENTENCE CHUNKING
# ============================================================

def split_sentences(
    text,
    target_chars=1800,
    max_chars=3200
):

    text = clean_text(text)

    if not text:

        return []

    if len(text) <= max_chars:

        return [text]

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    chunks = []

    current = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:

            continue

        proposed = (
            f"{current} {sentence}"
            .strip()
        )

        if (
            current
            and len(proposed) > target_chars
        ):

            chunks.append(
                current.strip()
            )

            current = sentence

        else:

            current = proposed

        # Handle very long individual sentences
        while len(current) > max_chars:

            chunks.append(
                current[:max_chars].strip()
            )

            current = current[max_chars:].strip()

    if current:

        chunks.append(
            current.strip()
        )

    return chunks


# ============================================================
# STRUCTURE-AWARE CHUNKING
# ============================================================

def make_chunks(
    elements,
    target_chars=1800,
    max_chars=3200
):

    chunks = []

    for element in elements:

        element_type = element.get(
            "type",
            "text"
        )

        page = element.get(
            "page"
        )

        content = clean_text(
            element.get(
                "content",
                ""
            )
        )

        if not content:

            continue

        # ----------------------------------------------------
        # Tables and visuals stay intact
        # ----------------------------------------------------

        if element_type in (
            "table",
            "visual"
        ):

            chunks.append(
                {
                    "text": content,
                    "type": element_type,
                    "page": page,
                }
            )

            continue

        # ----------------------------------------------------
        # Normal text
        # ----------------------------------------------------

        parts = split_sentences(
            content,
            target_chars=target_chars,
            max_chars=max_chars
        )

        for part in parts:

            chunks.append(
                {
                    "text": part,
                    "type": "text",
                    "page": page,
                }
            )

    return chunks


# ============================================================
# GEMINI ERROR HANDLING
# ============================================================

def classify_gemini_error(exc):

    message = str(exc).lower()

    if (
        "quota" in message
        or "429" in message
    ):

        return (
            "Gemini quota or rate limit was reached."
        )

    if (
        "503" in message
        or "unavailable" in message
    ):

        return (
            "The Gemini model is temporarily unavailable."
        )

    if (
        "404" in message
        or "not found" in message
    ):

        return (
            "The configured Gemini model "
            "is unavailable."
        )

    return str(exc)


# ============================================================
# VISUAL DESCRIPTION
# ============================================================

def describe_visual(image_bytes):

    client = get_gemini_client()

    prompt = """
You are analyzing a visual extracted from a PDF.

Describe ONLY information that is actually visible.

Include when available:

- title
- caption
- chart type
- labels
- axis information
- important values
- trends
- comparisons
- relationships
- categories
- other useful visual information

Do not invent values or facts.

Return a concise factual description suitable for
semantic document retrieval.
"""

    last_error = None

    for model in GEMINI_MODELS:

        try:

            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/png"
                    ),
                    prompt
                ]
            )

            text = getattr(
                response,
                "text",
                None
            )

            if text:

                return clean_text(text)

        except Exception as exc:

            last_error = exc

    if last_error:

        raise RuntimeError(
            classify_gemini_error(
                last_error
            )
        )

    return ""


# ============================================================
# EXTRACT PDF
# ============================================================

def extract_pdf(uploaded_file):

    suffix = (
        Path(
            uploaded_file.name
        ).suffix
        or ".pdf"
    )

    temp_path = None

    try:

        # ----------------------------------------------------
        # Save uploaded PDF temporarily
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_path = temp_file.name

        # ----------------------------------------------------
        # Docling
        # ----------------------------------------------------

        converter = get_converter()

        result = converter.convert(
            temp_path
        )

        doc = result.document

        elements = []

        table_count = 0
        visual_count = 0

        # ----------------------------------------------------
        # Iterate through document
        # ----------------------------------------------------

        try:

            for item, level in doc.iterate_items():

                item_class = (
                    item.__class__.__name__
                    .lower()
                )

                page = get_page_number(
                    item
                )

                # --------------------------------------------
                # TABLE
                # --------------------------------------------

                if "table" in item_class:

                    table_text = table_to_text(
                        item,
                        doc
                    )

                    if table_text:

                        elements.append(
                            {
                                "type": "table",
                                "page": page,
                                "content": table_text,
                            }
                        )

                        table_count += 1

                    continue

                # --------------------------------------------
                # PICTURE
                # --------------------------------------------

                if "picture" in item_class:

                    continue

                # --------------------------------------------
                # NORMAL TEXT
                # --------------------------------------------

                text = get_item_text(
                    item
                )

                if text:

                    elements.append(
                        {
                            "type": "text",
                            "page": page,
                            "content": text,
                        }
                    )

        except Exception as exc:

            # Don't silently destroy the entire PDF.
            # Preserve the exception so user gets useful info.
            raise RuntimeError(
                "Docling failed while extracting "
                f"document items: {exc}"
            ) from exc

        # ----------------------------------------------------
        # Process pictures separately
        # ----------------------------------------------------

        try:

            pictures = getattr(
                doc,
                "pictures",
                []
            )

            for picture in pictures:

                page = get_page_number(
                    picture
                )

                image_bytes = None

                # --------------------------------------------
                # Get image from Docling
                # --------------------------------------------

                try:

                    image = picture.get_image(
                        doc
                    )

                    if image is not None:

                        buffer = BytesIO()

                        image.save(
                            buffer,
                            format="PNG"
                        )

                        image_bytes = (
                            buffer.getvalue()
                        )

                except Exception:

                    image_bytes = None

                # --------------------------------------------
                # Gemini visual description
                # --------------------------------------------

                if image_bytes:

                    try:

                        description = describe_visual(
                            image_bytes
                        )

                    except Exception as exc:

                        description = (
                            "Visual description unavailable: "
                            + str(exc)
                        )

                    if description:

                        elements.append(
                            {
                                "type": "visual",
                                "page": page,
                                "content": description,
                            }
                        )

                        visual_count += 1

        except Exception:

            # Visual extraction failure should not destroy
            # otherwise usable text/table extraction.
            pass

        # ----------------------------------------------------
        # Sort by page
        # ----------------------------------------------------

        elements.sort(
            key=lambda item: (
                item.get("page")
                if item.get("page") is not None
                else 10**9
            )
        )

        # ----------------------------------------------------
        # Create chunks
        # ----------------------------------------------------

        chunks = make_chunks(
            elements
        )

        if not chunks:

            raise RuntimeError(
                "No searchable text, tables, or visual "
                "descriptions were extracted from the PDF."
            )

        # ----------------------------------------------------
        # Jina embeddings
        # ----------------------------------------------------

        embeddings = embed_documents(
            chunks
        )

        if embeddings.size == 0:

            raise RuntimeError(
                "Jina returned no embeddings."
            )

        # ----------------------------------------------------
        # Normalize embeddings
        # ----------------------------------------------------

        faiss.normalize_L2(
            embeddings
        )

        # ----------------------------------------------------
        # FAISS
        # ----------------------------------------------------

        index = faiss.IndexFlatIP(
            embeddings.shape[1]
        )

        index.add(
            embeddings
        )

        # ----------------------------------------------------
        # Number of pages
        # ----------------------------------------------------

        pages = 0

        try:

            pages = int(
                getattr(
                    doc,
                    "num_pages",
                    0
                )
                or 0
            )

        except Exception:

            pages = 0

        if not pages:

            page_numbers = [
                element["page"]
                for element in elements
                if element.get("page") is not None
            ]

            if page_numbers:

                pages = max(
                    page_numbers
                )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata = []

        for chunk in chunks:

            metadata.append(
                {
                    "page": chunk.get(
                        "page"
                    ),
                    "type": chunk.get(
                        "type",
                        "text"
                    ),
                }
            )

        stats = {
            "tables": table_count,
            "visuals": visual_count,
        }

        return {
            "chunks": chunks,
            "metadata": metadata,
            "index": index,
            "pages": pages,
            "stats": stats,
        }

    finally:

        # ----------------------------------------------------
        # Cleanup temporary PDF
        # ----------------------------------------------------

        if temp_path:

            try:

                os.remove(
                    temp_path
                )

            except Exception:

                pass


# ============================================================
# PUBLIC PROCESS FUNCTION
# ============================================================

def process_pdf(uploaded_file):

    return extract_pdf(
        uploaded_file
    )


# ============================================================
# JINA API
# ============================================================

def jina_request(
    texts,
    task
):

    if not texts:

        return []

    headers = {
        "Authorization": (
            f"Bearer {get_jina_key()}"
        ),
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
            "Jina Embeddings API error "
            f"{response.status_code}: "
            f"{response.text[:1000]}"
        )

    result = response.json()

    if "data" not in result:

        raise RuntimeError(
            "Unexpected response from Jina Embeddings API."
        )

    data = result["data"]

    data = sorted(
        data,
        key=lambda item: item["index"]
    )

    return [
        item["embedding"]
        for item in data
    ]


# ============================================================
# FORMAT DOCUMENT FOR EMBEDDING
# ============================================================

def format_document_for_embedding(
    chunk
):

    chunk_type = chunk.get(
        "type",
        "text"
    )

    page = chunk.get(
        "page"
    )

    prefix = ""

    if page is not None:

        prefix += (
            f"[Page {page}] "
        )

    if chunk_type == "table":

        prefix += "[TABLE] "

    elif chunk_type == "visual":

        prefix += "[VISUAL] "

    return (
        prefix
        + chunk.get(
            "text",
            ""
        )
    )


# ============================================================
# DOCUMENT EMBEDDINGS
# ============================================================

def embed_documents(
    chunks
):

    all_embeddings = []

    for start in range(
        0,
        len(chunks),
        JINA_BATCH_SIZE
    ):

        batch = chunks[
            start:start + JINA_BATCH_SIZE
        ]

        texts = [
            format_document_for_embedding(
                chunk
            )
            for chunk in batch
        ]

        embeddings = jina_request(
            texts,
            task="retrieval.passage"
        )

        all_embeddings.extend(
            embeddings
        )

    return np.asarray(
        all_embeddings,
        dtype="float32"
    )


# ============================================================
# QUERY EMBEDDING
# ============================================================

def embed_query(
    query
):

    embeddings = jina_request(
        [query],
        task="retrieval.query"
    )

    if not embeddings:

        raise RuntimeError(
            "Jina returned no query embedding."
        )

    return np.asarray(
        embeddings[0],
        dtype="float32"
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    question,
    index,
    chunks,
    metadata,
    top_k=5
):

    if index is None:

        return []

    if not chunks:

        return []

    query_vector = embed_query(
        question
    ).reshape(
        1,
        -1
    )

    faiss.normalize_L2(
        query_vector
    )

    number_to_retrieve = min(
        top_k,
        len(chunks)
    )

    scores, indices = index.search(
        query_vector,
        number_to_retrieve
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:

            continue

        if idx >= len(chunks):

            continue

        results.append(
            {
                "score": float(score),
                "chunk": chunks[idx],
                "metadata": metadata[idx],
            }
        )

    return results


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    results
):

    blocks = []

    for number, result in enumerate(
        results,
        start=1
    ):

        chunk = result[
            "chunk"
        ]

        metadata = result[
            "metadata"
        ]

        page = metadata.get(
            "page"
        )

        chunk_type = metadata.get(
            "type",
            "text"
        )

        if page is not None:

            source = (
                f"Page {page}"
            )

        else:

            source = "Page unknown"

        blocks.append(
            f"""
[Retrieved Source {number}]
Type: {chunk_type}
Source: {source}

{chunk.get("text", "")}
""".strip()
        )

    return "\n\n".join(
        blocks
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    results
):

    if not results:

        return (
            "I couldn't find relevant information "
            "in the document."
        )

    context = build_context(
        results
    )

    client = get_gemini_client()

    prompt = f"""
You are DocuMind, an intelligent PDF assistant.

Answer the user's question using ONLY the retrieved
document context provided below.

STRICT RULES:

1. Use only the retrieved document context.
2. Do not use outside knowledge.
3. Do not invent facts, numbers, dates, names or conclusions.
4. If the retrieved context does not contain enough information,
   clearly say that the document context does not provide enough
   information.
5. Mention page numbers when useful.
6. For tables, use the values actually present in the table.
7. For charts or visuals, use only information contained in the
   visual description.
8. Keep the answer clear and reasonably concise.

RETRIEVED DOCUMENT CONTEXT
==========================

{context}

USER QUESTION
=============

{question}
"""

    last_error = None

    for model in GEMINI_MODELS:

        try:

            response = client.models.generate_content(
                model=model,
                contents=prompt
            )

            text = getattr(
                response,
                "text",
                None
            )

            if text:

                return text.strip()

        except Exception as exc:

            last_error = exc

    if last_error:

        raise RuntimeError(
            classify_gemini_error(
                last_error
            )
        )

    return (
        "I could not generate an answer "
        "from the retrieved document context."
    )


# ============================================================
# CACHE CLEANUP
# ============================================================

def clear_backend_cache():

    # We intentionally keep expensive cached clients.
    # The active PDF/index is stored in session_state.
    return None
