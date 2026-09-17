import os
import re
import tempfile
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import requests
import streamlit as st

from google import genai
from google.genai import types

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)
from docling.datamodel.base_models import (
    InputFormat,
)


# ============================================================
# CONFIGURATION
# ============================================================

JINA_EMBEDDING_URL = (
    "https://api.jina.ai/v1/embeddings"
)

JINA_MODEL = "jina-embeddings-v4"

GEMINI_TEXT_MODEL = "gemini-2.5-flash"

GEMINI_VISION_MODEL = "gemini-2.5-flash"

TOP_K = 5

CHUNK_SIZE = 900

CHUNK_OVERLAP = 150


# ============================================================
# API KEYS
# ============================================================

def get_secret(name: str):

    value = None

    try:
        value = st.secrets.get(name)
    except Exception:
        pass

    if not value:
        value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"{name} is not configured."
        )

    return value


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_gemini_client():

    api_key = get_secret(
        "GEMINI_API_KEY"
    )

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# DOCLING CONVERTER
# ============================================================

@st.cache_resource
def get_document_converter():

    pipeline_options = PdfPipelineOptions()

    # --------------------------------------------------------
    # IMPORTANT:
    # Disable OCR.
    #
    # This prevents RapidOCR model permission/loading issues
    # on Streamlit Cloud.
    # --------------------------------------------------------

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
# TEXT CLEANING
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


# ============================================================
# PAGE NUMBER
# ============================================================

def get_page_number(item):

    try:

        prov = getattr(
            item,
            "prov",
            None
        )

        if prov:

            first = prov[0]

            page_no = getattr(
                first,
                "page_no",
                None
            )

            if page_no is not None:
                return int(page_no)

    except Exception:
        pass

    return None


# ============================================================
# GENERIC ITEM TEXT
# ============================================================

def get_item_text(item):

    try:

        text = getattr(
            item,
            "text",
            None
        )

        if text:
            return clean_text(text)

    except Exception:
        pass

    try:

        text = getattr(
            item,
            "content",
            None
        )

        if text:
            return clean_text(text)

    except Exception:
        pass

    return ""


# ============================================================
# TABLE TO TEXT
# ============================================================

def table_to_text(
    table,
    doc
):

    # --------------------------------------------------------
    # Preferred:
    # Export dataframe WITH doc argument.
    # --------------------------------------------------------

    try:

        dataframe = table.export_to_dataframe(
            doc=doc
        )

        if isinstance(
            dataframe,
            pd.DataFrame
        ):

            if not dataframe.empty:

                return clean_text(
                    dataframe.to_markdown(
                        index=False
                    )
                )

    except Exception:
        pass


    # --------------------------------------------------------
    # Fallback:
    # export markdown WITH doc argument.
    # --------------------------------------------------------

    try:

        markdown = table.export_to_markdown(
            doc=doc
        )

        if markdown:
            return clean_text(
                markdown
            )

    except Exception:
        pass


    return ""


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text):

    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        s.strip()
        for s in sentences
        if s.strip()
    ]


# ============================================================
# CHUNK CREATION
# ============================================================

def make_chunks(
    text,
    page=None,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):

    text = clean_text(text)

    if not text:
        return []

    sentences = split_sentences(
        text
    )

    chunks = []

    current = ""

    for sentence in sentences:

        candidate = (
            current + " " + sentence
        ).strip()

        if len(candidate) <= chunk_size:

            current = candidate

        else:

            if current:

                chunks.append(
                    {
                        "text": current,
                        "page": page,
                    }
                )

            if overlap > 0:

                overlap_text = (
                    current[-overlap:]
                    if current
                    else ""
                )

                current = (
                    overlap_text
                    + " "
                    + sentence
                ).strip()

            else:

                current = sentence


    if current:

        chunks.append(
            {
                "text": current,
                "page": page,
            }
        )


    return chunks


# ============================================================
# VISUAL DESCRIPTION
# ============================================================

def describe_visual(
    picture,
    doc,
):

    try:

        image = picture.get_image(
            doc
        )

        if image is None:
            return ""


        # Convert image to PNG bytes
        image_buffer = tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        )

        try:

            image.save(
                image_buffer.name,
                format="PNG"
            )

            with open(
                image_buffer.name,
                "rb"
            ) as f:

                image_bytes = f.read()

        finally:

            try:
                os.unlink(
                    image_buffer.name
                )
            except Exception:
                pass


        client = get_gemini_client()


        response = client.models.generate_content(
            model=GEMINI_VISION_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png",
                ),
                (
                    "Describe this visual from a PDF "
                    "in a concise factual way. "
                    "Include important labels, values, "
                    "relationships, trends, categories, "
                    "and other information that would "
                    "help answer questions about it."
                ),
            ],
        )


        if response and response.text:

            return clean_text(
                response.text
            )


    except Exception:

        # Visual processing failure should not
        # stop the entire PDF pipeline.
        pass


    return ""


# ============================================================
# JINA EMBEDDINGS
# ============================================================

def jina_request(
    texts,
):

    api_key = get_secret(
        "JINA_API_KEY"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": JINA_MODEL,
        "input": texts,
        "encoding_type": "float",
    }

    response = requests.post(
        JINA_EMBEDDING_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    result = response.json()

    data = result.get(
        "data",
        []
    )

    embeddings = [
        item["embedding"]
        for item in data
    ]

    return np.asarray(
        embeddings,
        dtype="float32"
    )


# ============================================================
# DOCUMENT EMBEDDINGS
# ============================================================

def embed_documents(
    texts,
    batch_size=32,
):

    all_embeddings = []

    for start in range(
        0,
        len(texts),
        batch_size,
    ):

        batch = texts[
            start:start + batch_size
        ]

        embeddings = jina_request(
            batch
        )

        all_embeddings.append(
            embeddings
        )


    if not all_embeddings:

        return np.empty(
            (0, 0),
            dtype="float32"
        )


    return np.vstack(
        all_embeddings
    )


# ============================================================
# QUERY EMBEDDING
# ============================================================

def embed_query(
    query
):

    embedding = jina_request(
        [query]
    )

    return embedding[0]


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(
    uploaded_file
):

    converter = get_document_converter()


    # --------------------------------------------------------
    # Save uploaded PDF temporarily
    # --------------------------------------------------------

    suffix = Path(
        uploaded_file.name
    ).suffix or ".pdf"

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        temp_path = temp_file.name


    try:

        # ----------------------------------------------------
        # Convert PDF
        # ----------------------------------------------------

        conversion_result = converter.convert(
            temp_path
        )

        doc = conversion_result.document


        # ----------------------------------------------------
        # Get page count
        # ----------------------------------------------------

        try:

            pages = len(
                doc.pages
            )

        except Exception:

            pages = 0


        # ----------------------------------------------------
        # Collect structured content
        # ----------------------------------------------------

        raw_items = []

        table_count = 0
        visual_count = 0


        for item, level in doc.iterate_items():

            page = get_page_number(
                item
            )


            # ------------------------------------------------
            # TABLE
            # ------------------------------------------------

            item_type = type(
                item
            ).__name__.lower()


            if "table" in item_type:

                table_text = table_to_text(
                    item,
                    doc
                )

                if table_text:

                    raw_items.append(
                        {
                            "text": (
                                "TABLE:\n"
                                + table_text
                            ),
                            "page": page,
                            "type": "table",
                        }
                    )

                    table_count += 1

                continue


            # ------------------------------------------------
            # PICTURE / VISUAL
            # ------------------------------------------------

            if "picture" in item_type:

                description = describe_visual(
                    item,
                    doc
                )

                if description:

                    raw_items.append(
                        {
                            "text": (
                                "VISUAL DESCRIPTION:\n"
                                + description
                            ),
                            "page": page,
                            "type": "visual",
                        }
                    )

                    visual_count += 1

                continue


            # ------------------------------------------------
            # NORMAL TEXT
            # ------------------------------------------------

            text = get_item_text(
                item
            )

            if text:

                raw_items.append(
                    {
                        "text": text,
                        "page": page,
                        "type": "text",
                    }
                )


        # ----------------------------------------------------
        # Create chunks
        # ----------------------------------------------------

        chunks = []


        for item in raw_items:

            item_chunks = make_chunks(
                item["text"],
                page=item["page"],
            )

            for chunk in item_chunks:

                chunk["type"] = item[
                    "type"
                ]

                chunks.append(
                    chunk
                )


        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if not chunks:

            raise RuntimeError(
                "No searchable content could be extracted from this PDF."
            )


        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = embed_documents(
            texts
        )


        if len(embeddings) == 0:

            raise RuntimeError(
                "No embeddings were generated."
            )


        # ----------------------------------------------------
        # FAISS
        #
        # Normalize vectors so inner product corresponds
        # to cosine similarity.
        # ----------------------------------------------------

        faiss.normalize_L2(
            embeddings
        )

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(
            dimension
        )

        index.add(
            embeddings
        )


        # ----------------------------------------------------
        # Return document knowledge base
        # ----------------------------------------------------

        return {
            "pages": pages,
            "chunks": len(chunks),
            "tables": table_count,
            "visuals": visual_count,
            "chunks_data": chunks,
            "embeddings": embeddings,
            "index": index,
        }


    finally:

        try:

            os.unlink(
                temp_path
            )

        except Exception:
            pass


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    query,
    document_data,
    top_k=TOP_K,
):

    index = document_data[
        "index"
    ]

    chunks = document_data[
        "chunks_data"
    ]


    query_embedding = embed_query(
        query
    )

    query_embedding = (
        query_embedding
        .reshape(1, -1)
        .astype("float32")
    )


    faiss.normalize_L2(
        query_embedding
    )


    k = min(
        top_k,
        index.ntotal
    )


    scores, indices = index.search(
        query_embedding,
        k
    )


    results = []


    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        chunk = chunks[idx].copy()

        chunk["score"] = float(
            score
        )

        results.append(
            chunk
        )


    return results


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    retrieved_chunks
):

    context_parts = []


    for i, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):

        page = chunk.get(
            "page"
        )

        chunk_type = chunk.get(
            "type",
            "text"
        )

        text = chunk.get(
            "text",
            ""
        )


        if page is not None:

            header = (
                f"[Retrieved Chunk {i} | "
                f"Page {page} | "
                f"{chunk_type.upper()}]"
            )

        else:

            header = (
                f"[Retrieved Chunk {i} | "
                f"{chunk_type.upper()}]"
            )


        context_parts.append(
            header
            + "\n"
            + text
        )


    return "\n\n".join(
        context_parts
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    context,
):

    client = get_gemini_client()


    prompt = f"""
You are DocuMind, an intelligent document
question-answering assistant.

Answer the user's question using ONLY the
provided document context.

DOCUMENT CONTEXT:
-----------------
{context}
-----------------

USER QUESTION:
{question}

Instructions:

1. Answer directly and clearly.
2. Use only information supported by the context.
3. Do not invent facts.
4. If the context does not contain enough information,
   clearly say that the information is not available
   in the retrieved document content.
5. When useful, mention page numbers.
6. For tables, preserve important values and relationships.
7. For charts or visuals, explain the information described
   in the retrieved visual context.
8. Keep the answer reasonably concise but informative.
"""


    response = client.models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=prompt,
    )


    if not response:

        return (
            "I couldn't generate an answer "
            "from the document."
        )


    if not response.text:

        return (
            "I couldn't generate an answer "
            "from the retrieved document context."
        )


    return response.text.strip()


# ============================================================
# CLEAR CACHE
# ============================================================

def clear_backend_cache():

    # Streamlit resource caches
    # will be recreated when needed.

    try:

        st.cache_resource.clear()

    except Exception:
        pass
