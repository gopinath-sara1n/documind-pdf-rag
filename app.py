import os
import re
import time
import random
import tempfile
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
import requests

from google import genai
from google.genai import types

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="DocuMind — Intelligent PDF Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
<style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .block-container {
        max-width: 1280px;
        padding-top: 1.2rem;
        padding-bottom: 4rem;
    }

    /* ========================================================
       TOP BRAND HEADER
       ======================================================== */

    .brand-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }

    .brand-left {
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    .brand-icon {
        width: 48px;
        height: 48px;
        border-radius: 15px;
        background: linear-gradient(
            135deg,
            #6366f1,
            #8b5cf6,
            #ec4899
        );
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.55rem;
        box-shadow: 0 8px 25px rgba(99,102,241,.25);
    }

    .brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .brand-subtitle {
        font-size: .78rem;
        opacity: .62;
        margin-top: .15rem;
    }

    .status-pill {
        padding: .45rem .8rem;
        border-radius: 999px;
        font-size: .75rem;
        font-weight: 700;
        background: rgba(16,185,129,.10);
        color: #059669;
        border: 1px solid rgba(16,185,129,.18);
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.1rem;
        border-radius: 25px;
        margin-bottom: 1.25rem;

        background:
            radial-gradient(
                circle at 90% 15%,
                rgba(236,72,153,.45),
                transparent 30%
            ),
            radial-gradient(
                circle at 70% 100%,
                rgba(139,92,246,.38),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #312e81 0%,
                #4f46e5 40%,
                #7c3aed 72%,
                #be185d 100%
            );

        color: white;
        box-shadow: 0 18px 45px rgba(79,70,229,.20);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.45rem;
        font-weight: 850;
        letter-spacing: -0.045em;
    }

    .hero p {
        margin: .65rem 0 0 0;
        color: rgba(255,255,255,.86);
        font-size: 1.02rem;
        max-width: 760px;
        line-height: 1.55;
    }

    .hero-badge {
        display: inline-block;
        padding: .35rem .65rem;
        border-radius: 999px;
        background: rgba(255,255,255,.14);
        border: 1px solid rgba(255,255,255,.18);
        font-size: .72rem;
        font-weight: 700;
        margin-bottom: .75rem;
    }


    /* ========================================================
       TABS
       ======================================================== */

    button[data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 700;
        padding: .75rem 1.35rem;
    }

    div[data-baseweb="tab-list"] {
        gap: .45rem;
        margin-bottom: 1.5rem;
    }


    /* ========================================================
       CARDS
       ======================================================== */

    .card {
        padding: 1.15rem 1.25rem;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,.18);
        background: rgba(128,128,128,.035);
        margin-bottom: 1rem;
    }

    .gradient-card {
        padding: 1.35rem;
        border-radius: 20px;
        color: white;
        background:
            linear-gradient(
                135deg,
                #4f46e5,
                #7c3aed 55%,
                #db2777
            );
        box-shadow: 0 12px 35px rgba(99,102,241,.18);
        margin-bottom: 1rem;
    }

    .gradient-card h3 {
        margin: 0 0 .4rem 0;
    }

    .gradient-card p {
        margin: 0;
        opacity: .88;
        line-height: 1.5;
    }


    /* ========================================================
       UPLOAD CARD
       ======================================================== */

    .upload-card {
        padding: 1.5rem;
        border-radius: 22px;
        border: 2px dashed rgba(99,102,241,.35);
        background:
            linear-gradient(
                180deg,
                rgba(99,102,241,.055),
                rgba(236,72,153,.035)
            );
        text-align: center;
        margin-bottom: 1rem;
    }

    .upload-icon {
        font-size: 2.4rem;
        margin-bottom: .35rem;
    }

    .upload-title {
        font-size: 1.15rem;
        font-weight: 800;
    }

    .upload-text {
        font-size: .88rem;
        opacity: .65;
        margin-top: .3rem;
    }


    /* ========================================================
       METRICS
       ======================================================== */

    .metric-card {
        padding: 1rem .8rem;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,.17);
        background: rgba(128,128,128,.035);
        text-align: center;
        min-height: 112px;
    }

    .metric-icon {
        font-size: 1.2rem;
    }

    .metric-value {
        font-size: 1.55rem;
        font-weight: 800;
        margin-top: .15rem;
    }

    .metric-label {
        font-size: .76rem;
        opacity: .62;
        margin-top: .05rem;
    }


    /* ========================================================
       DOCUMENT READY
       ======================================================== */

    .document-ready {
        padding: .85rem 1rem;
        border-radius: 16px;
        background: rgba(16,185,129,.075);
        border: 1px solid rgba(16,185,129,.18);
        margin-bottom: 1rem;
    }

    .document-name {
        font-weight: 800;
    }

    .document-status {
        font-size: .78rem;
        opacity: .65;
        margin-top: .2rem;
    }


    /* ========================================================
       SUGGESTED QUESTIONS
       ======================================================== */

    .section-label {
        font-size: .8rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .07em;
        opacity: .58;
        margin: 1.2rem 0 .65rem 0;
    }


    /* ========================================================
       PIPELINE
       ======================================================== */

    .pipeline {
        display: flex;
        flex-wrap: wrap;
        gap: .45rem;
        margin-top: .7rem;
    }

    .pipeline-step {
        padding: .42rem .7rem;
        border-radius: 999px;
        font-size: .73rem;
        font-weight: 700;
        background: rgba(99,102,241,.09);
        border: 1px solid rgba(99,102,241,.13);
    }


    /* ========================================================
       ABOUT CARDS
       ======================================================== */

    .about-card {
        padding: 1.25rem;
        border-radius: 19px;
        border: 1px solid rgba(128,128,128,.17);
        background: rgba(128,128,128,.035);
        height: 100%;
    }

    .about-icon {
        font-size: 1.65rem;
        margin-bottom: .45rem;
    }

    .about-title {
        font-weight: 800;
        font-size: 1rem;
        margin-bottom: .35rem;
    }

    .about-text {
        font-size: .86rem;
        line-height: 1.55;
        opacity: .68;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 2.7rem;
    }


    /* ========================================================
       CHAT
       ======================================================== */

    div[data-testid="stChatMessage"] {
        border-radius: 16px;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(128,128,128,.15);
        font-size: .75rem;
        opacity: .5;
    }

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "ready": False,
    "file_name": None,
    "pages": 0,
    "chunks": [],
    "metadata": [],
    "index": None,
    "messages": [],
    "stats": {},
    "pending_question": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CONFIG
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
# API CLIENTS
# ============================================================

def get_jina_key():
    try:
        return st.secrets["JINA_API_KEY"]
    except Exception:
        return os.getenv("JINA_API_KEY")


def get_gemini_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY")


# ============================================================
# DOCLING / TEXT HELPERS
# ============================================================

def clean_text(text):
    if text is None:
        return ""

    text = str(text).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def get_item_text(item):

    text = getattr(item, "text", None)

    if isinstance(text, str) and text.strip():
        return clean_text(text)

    orig = getattr(item, "orig", None)

    if isinstance(orig, str) and orig.strip():
        return clean_text(orig)

    if isinstance(item, dict):

        for key in (
            "text",
            "orig",
            "content"
        ):

            value = item.get(key)

            if isinstance(value, str) and value.strip():
                return clean_text(value)

    return ""


def get_page_number(item):

    try:

        prov = getattr(item, "prov", [])

        if prov:

            page_no = getattr(
                prov[0],
                "page_no",
                None
            )

            if page_no is None and isinstance(
                prov[0],
                dict
            ):

                page_no = prov[0].get(
                    "page_no"
                )

            return page_no

    except Exception:
        pass

    return None


def table_to_text(table):

    try:

        df = table.export_to_dataframe()

        if df is not None:

            df = df.fillna("")

            lines = []

            columns = [
                clean_text(x)
                for x in df.columns
            ]

            if any(columns):
                lines.append(
                    " | ".join(columns)
                )

            for _, row in df.iterrows():

                values = [
                    clean_text(v)
                    for v in row.tolist()
                ]

                if any(values):

                    lines.append(
                        " | ".join(
                            v if v else "-"
                            for v in values
                        )
                    )

            result = "\n".join(lines).strip()

            if result:
                return result

    except Exception:
        pass

    return get_item_text(table)


def split_sentences(text):

    text = clean_text(text)

    if not text:
        return []

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    sentences = []

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        parts = re.split(
            r"(?<=[.!?])\s+(?=[A-Z₹0-9(\[])",
            paragraph,
        )

        for part in parts:

            part = part.strip()

            if part:
                sentences.append(part)

    return sentences


# ============================================================
# STAGE 1 — DOCLING
# ============================================================

@st.cache_resource(show_spinner=False)
def get_converter():

    pipeline_options = PdfPipelineOptions()

    pipeline_options.do_table_structure = True

    pipeline_options.generate_picture_images = True

    pipeline_options.do_picture_description = False

    # IMPORTANT:
    # Keep OCR disabled because the Streamlit deployment
    # environment caused RapidOCR model permission issues.
    pipeline_options.do_ocr = False

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )


def extract_pdf(
    pdf_bytes,
    file_name,
    progress
):

    with tempfile.TemporaryDirectory() as temp_dir:

        pdf_path = Path(temp_dir) / file_name

        pdf_path.write_bytes(pdf_bytes)

        progress.progress(
            0.12,
            "Reading PDF with Docling…"
        )

        result = get_converter().convert(
            str(pdf_path)
        )

        doc = result.document

        elements = []

        text_count = 0
        table_count = 0
        visual_count = 0

        # Preserve document order.
        for position, (
            item,
            level
        ) in enumerate(
            doc.iterate_items()
        ):

            page = get_page_number(item)

            item_name = type(
                item
            ).__name__.lower()

            # -----------------------------
            # TABLE
            # -----------------------------

            if "table" in item_name:

                content = table_to_text(
                    item
                )

                if content:

                    elements.append({

                        "type": "table",

                        "page": page,

                        "position": position,

                        "content": content,

                        "label": "table",

                    })

                    table_count += 1

                continue

            # -----------------------------
            # PICTURE
            # -----------------------------

            if "picture" in item_name:

                continue

            # -----------------------------
            # TEXT
            # -----------------------------

            content = get_item_text(item)

            if not content:
                continue

            label = getattr(
                item,
                "label",
                "text"
            )

            elements.append({

                "type": "text",

                "page": page,

                "position": position,

                "content": content,

                "label": str(label),

            })

            text_count += 1

        progress.progress(
            0.35,
            "Extracting text and tables…"
        )

        # ====================================================
        # VISUAL UNDERSTANDING
        # ====================================================

        pictures = list(
            getattr(
                doc,
                "pictures",
                []
            )
        )

        visual_candidates = []

        for idx, picture in enumerate(
            pictures,
            start=1
        ):

            try:

                image = picture.get_image(
                    doc
                )

                if image is None:
                    continue

                width, height = image.size

                if width < 200 or height < 80:
                    continue

                page = get_page_number(
                    picture
                )

                visual_candidates.append(
                    (
                        idx,
                        page,
                        image
                    )
                )

            except Exception:
                continue

        visual_total = len(
            visual_candidates
        )

        if visual_total:

            client = genai.Client(
                api_key=get_gemini_key()
            )

            for n, (
                idx,
                page,
                image
            ) in enumerate(
                visual_candidates,
                start=1
            ):

                description = describe_visual(
                    client,
                    image,
                    page,
                    GEMINI_MODELS
                )

                if description:

                    elements.append({

                        "type": "visual",

                        "page": page,

                        "position":
                            10_000_000 + idx,

                        "content":
                            description,

                        "label":
                            "visual",

                    })

                    visual_count += 1

                progress.progress(
                    0.35
                    + (
                        0.25
                        * n
                        / max(
                            visual_total,
                            1
                        )
                    ),
                    f"Understanding visuals "
                    f"({n}/{visual_total})…"
                )

        elements.sort(
            key=lambda x: x["position"]
        )

        progress.progress(
            0.62,
            "Building structure-aware chunks…"
        )

        chunks = make_chunks(
            elements
        )

        progress.progress(
            0.66,
            "Creating Jina embeddings…"
        )

        embeddings = embed_documents(
            [
                format_document_for_embedding(c)
                for c in chunks
            ],
            progress
        )

        index = faiss.IndexFlatIP(
            JINA_DIMENSIONS
        )

        index.add(
            embeddings
        )

        metadata = [

            {

                "chunk_id":
                    c["chunk_id"],

                "page_start":
                    c.get("page_start"),

                "page_end":
                    c.get("page_end"),

                "section":
                    c.get("section", ""),

                "element_types":
                    c.get(
                        "element_types",
                        []
                    ),

                "has_table":
                    c.get(
                        "has_table",
                        False
                    ),

                "has_visual":
                    c.get(
                        "has_visual",
                        False
                    ),

            }

            for c in chunks

        ]

        progress.progress(
            1.0,
            "Document is ready."
        )

        return {

            "chunks": chunks,

            "metadata": metadata,

            "index": index,

            "pages":
                len(
                    getattr(
                        doc,
                        "pages",
                        {}
                    ) or {}
                ),

            "stats": {

                "text":
                    text_count,

                "tables":
                    table_count,

                "visuals":
                    visual_count,

                "chunks":
                    len(chunks),

            },

        }


# ============================================================
# GEMINI VISUAL UNDERSTANDING
# ============================================================

def classify_gemini_error(exc):

    text = str(exc).upper()

    if any(
        x in text
        for x in [
            "401",
            "403",
            "UNAUTHENTICATED",
            "PERMISSION_DENIED"
        ]
    ):
        return "auth"

    if any(
        x in text
        for x in [
            "429",
            "RESOURCE_EXHAUSTED",
            "RATE LIMIT",
            "QUOTA"
        ]
    ):
        return "quota"

    if any(
        x in text
        for x in [
            "503",
            "UNAVAILABLE",
            "500",
            "502",
            "504",
            "INTERNAL"
        ]
    ):
        return "temporary"

    return "permanent"


def describe_visual(
    client,
    image,
    page,
    models
):

    prompt = f"""
You are analyzing a visual extracted from page {page or 'unknown'}
of a complex PDF document for a Retrieval-Augmented Generation system.

Create a highly accurate, self-contained description.

Include, when visible:
1. Chart or figure title.
2. Type of visual.
3. Important categories, labels and legends.
4. Important numerical values.
5. Dates, years or periods.
6. Units.
7. Trends, comparisons and relationships.
8. Forecast values, if present.
9. Important annotations.
10. Source, if visible.
11. Meaning directly supported by the visual.

Do NOT invent information that is not visible.
Do NOT provide opinions or recommendations.
Preserve exact terminology, names and numerical values whenever readable.

Return ONLY the description.
""".strip()

    import io

    buf = io.BytesIO()

    image.save(
        buf,
        format="PNG"
    )

    image_bytes = buf.getvalue()

    for model in models:

        for attempt in range(3):

            try:

                response = client.models.generate_content(

                    model=model,

                    contents=[

                        types.Part.from_bytes(

                            data=image_bytes,

                            mime_type="image/png",

                        ),

                        prompt,

                    ],

                )

                if response.text:

                    return response.text.strip()

            except Exception as exc:

                error_type = classify_gemini_error(
                    exc
                )

                if error_type in {
                    "auth",
                    "permanent"
                }:
                    break

                if attempt < 2:

                    time.sleep(
                        min(
                            2 ** attempt
                            + random.random(),
                            8
                        )
                    )

    return ""


# ============================================================
# STAGE 2 — STRUCTURE-AWARE CHUNKING
# ============================================================

def make_chunks(
    elements,
    target_chars=1800,
    max_chars=3200
):

    chunks = []

    current_parts = []
    current_ids = []
    current_types = []
    current_pages = []

    current_section = ""

    def current_text():

        return "\n\n".join(
            current_parts
        ).strip()

    def flush():

        nonlocal current_parts
        nonlocal current_ids
        nonlocal current_types
        nonlocal current_pages

        text = current_text()

        if not text:

            current_parts = []
            current_ids = []
            current_types = []
            current_pages = []

            return

        chunks.append({

            "chunk_id":
                f"chunk_{len(chunks):06d}",

            "page_start":
                min(current_pages)
                if current_pages
                else None,

            "page_end":
                max(current_pages)
                if current_pages
                else None,

            "section":
                current_section,

            "content":
                text,

            "element_ids":
                list(current_ids),

            "element_types":
                list(current_types),

            "has_table":
                "table"
                in current_types,

            "has_visual":
                "visual"
                in current_types,

        })

        current_parts = []
        current_ids = []
        current_types = []
        current_pages = []

    def add(
        content,
        element
    ):

        content = clean_text(
            content
        )

        if not content:
            return

        current_parts.append(
            content
        )

        current_ids.append(
            f"element_"
            f"{element.get('position', len(current_ids)):06d}"
        )

        current_types.append(
            element["type"]
        )

        if element.get("page") is not None:

            current_pages.append(
                element["page"]
            )

    for element in elements:

        kind = element["type"]

        label = element.get(
            "label",
            ""
        )

        content = element["content"]

        # -----------------------------
        # SECTION HEADER
        # -----------------------------

        if (
            kind == "text"
            and label == "section_header"
        ):

            flush()

            current_section = content

            add(
                content,
                element
            )

            continue

        # -----------------------------
        # TABLE / VISUAL
        # Keep intact
        # -----------------------------

        if kind in {
            "table",
            "visual"
        }:

            if current_text():

                candidate = (
                    current_text()
                    + "\n\n"
                    + content
                )

                if len(candidate) > target_chars:

                    flush()

            add(
                content,
                element
            )

            continue

        # -----------------------------
        # TEXT
        # -----------------------------

        for sentence in split_sentences(
            content
        ):

            candidate = (

                current_text()
                + "\n\n"
                + sentence

                if current_text()

                else sentence

            )

            if len(candidate) <= target_chars:

                add(
                    sentence,
                    element
                )

                continue

            flush()

            if len(sentence) <= max_chars:

                add(
                    sentence,
                    element
                )

            else:

                words = sentence.split()

                buffer = []

                for word in words:

                    test = " ".join(
                        buffer + [word]
                    )

                    if len(test) <= max_chars:

                        buffer.append(
                            word
                        )

                    else:

                        if buffer:

                            add(
                                " ".join(buffer),
                                element
                            )

                            flush()

                        buffer = [word]

                if buffer:

                    add(
                        " ".join(buffer),
                        element
                    )

    flush()

    return chunks


# ============================================================
# STAGE 3 — JINA EMBEDDING
# ============================================================

def jina_request(
    inputs,
    task
):

    key = get_jina_key()

    if not key:

        raise RuntimeError(
            "JINA_API_KEY is missing. "
            "Add it to Streamlit Secrets."
        )

    headers = {

        "Authorization":
            f"Bearer {key}",

        "Content-Type":
            "application/json",

        "Accept":
            "application/json",

    }

    payload = {

        "model":
            JINA_MODEL,

        "input":
            inputs,

        "task":
            task,

        "embedding_type":
            "float",

        "dimensions":
            JINA_DIMENSIONS,

        "truncate":
            False,

    }

    last_error = None

    for attempt in range(6):

        try:

            response = requests.post(

                JINA_ENDPOINT,

                headers=headers,

                json=payload,

                timeout=180,

            )

            if response.status_code != 200:

                raise RuntimeError(

                    f"Jina HTTP "
                    f"{response.status_code}: "
                    f"{response.text[:400]}"

                )

            data = response.json().get(
                "data",
                []
            )

            data = sorted(
                data,
                key=lambda x: x["index"]
            )

            vectors = np.asarray(

                [
                    x["embedding"]
                    for x in data
                ],

                dtype="float32",

            )

            if len(vectors) != len(inputs):

                raise RuntimeError(
                    "Jina returned an "
                    "unexpected number "
                    "of embeddings."
                )

            vectors /= np.maximum(

                np.linalg.norm(
                    vectors,
                    axis=1,
                    keepdims=True
                ),

                1e-12

            )

            return vectors

        except Exception as exc:

            last_error = exc

            if attempt == 5:
                break

            time.sleep(

                min(
                    2 ** attempt
                    + random.random(),
                    20
                )

            )

    raise last_error


def embed_documents(
    texts,
    progress
):

    all_vectors = []

    total = len(texts)

    for start in range(
        0,
        total,
        JINA_BATCH_SIZE
    ):

        batch = texts[
            start:
            start + JINA_BATCH_SIZE
        ]

        vectors = jina_request(
            batch,
            "retrieval.passage"
        )

        all_vectors.append(
            vectors
        )

        done = min(
            start + len(batch),
            total
        )

        progress.progress(

            0.66
            + (
                0.25
                * done
                / max(total, 1)
            ),

            f"Embedding chunks "
            f"({done}/{total})…"

        )

    return np.vstack(
        all_vectors
    ).astype("float32")


def embed_query(question):

    return jina_request(
        [question],
        "retrieval.query"
    )


def format_document_for_embedding(
    chunk
):

    title = (
        chunk.get("section")
        or "none"
    )

    return (
        f"title: {title} | "
        f"text: {chunk['content']}"
    )


# ============================================================
# STAGE 4 — RETRIEVAL
# ============================================================

def retrieve(
    question,
    top_k=5
):

    q = embed_query(
        question
    )

    scores, indices = (
        st.session_state.index.search(
            q,
            top_k
        )
    )

    results = []

    for rank, (
        score,
        idx
    ) in enumerate(
        zip(
            scores[0],
            indices[0]
        ),
        start=1
    ):

        if idx < 0:
            continue

        results.append({

            "rank":
                rank,

            "score":
                float(score),

            "chunk":
                st.session_state.chunks[
                    idx
                ],

            "metadata":
                st.session_state.metadata[
                    idx
                ],

        })

    return results


def build_context(
    results
):

    parts = []

    for item in results:

        chunk = item["chunk"]

        page = chunk.get(
            "page_start"
        )

        page_end = chunk.get(
            "page_end"
        )

        page_text = (

            f"page {page}"

            if (
                page == page_end
                or page_end is None
            )

            else
            f"pages {page}-{page_end}"

        )

        parts.append(

            f"[Source {item['rank']} | "
            f"{page_text}]\n"
            f"{chunk['content']}"

        )

    return "\n\n".join(
        parts
    )


# ============================================================
# GEMINI ANSWER
# ============================================================

def generate_answer(
    question,
    results
):

    key = get_gemini_key()

    if not key:

        raise RuntimeError(
            "GEMINI_API_KEY is missing. "
            "Add it to Streamlit Secrets."
        )

    client = genai.Client(
        api_key=key
    )

    context = build_context(
        results
    )

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the retrieved document
context below.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. Compare the retrieved sources before answering.
4. Prefer the source that directly answers the exact question.
5. Pay close attention to dates, categories, units, metrics,
   geographic scope and time periods.
6. Preserve numerical values from the document.
7. If calculation is required, calculate only from numbers
   provided in the context.
8. If the context is insufficient, clearly say the information
   is not available in the retrieved document.
9. For multi-part questions, answer every part.
10. Mention page number(s) when useful.
11. Do not mention FAISS, embeddings, chunks or retrieval unless
    the user specifically asks about the system.
12. Use clean Markdown. Do not use LaTeX.

RETRIEVED DOCUMENT CONTEXT
==========================
{context}

USER QUESTION
=============
{question}

FINAL ANSWER
============
""".strip()

    last_error = None

    for model in GEMINI_MODELS:

        try:

            response = client.models.generate_content(

                model=model,

                contents=prompt,

            )

            if response.text:

                return (
                    response.text.strip(),
                    model
                )

        except Exception as exc:

            last_error = exc

            time.sleep(1)

    raise RuntimeError(
        f"Gemini could not generate "
        f"an answer: {last_error}"
    )


# ============================================================
# CLEAR DOCUMENT
# ============================================================

def clear_document():

    for key, value in DEFAULT_STATE.items():

        st.session_state[key] = value

    st.rerun()


# ============================================================
# BRAND HEADER
# ============================================================

st.markdown(
    """
<div class="brand-row">

    <div class="brand-left">

        <div class="brand-icon">
            📄
        </div>

        <div>

            <div class="brand-title">
                DocuMind
            </div>

            <div class="brand-subtitle">
                Intelligent PDF Assistant
            </div>

        </div>

    </div>

    <div class="status-pill">
        ✦ AI-Powered RAG
    </div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# TOP TABS
# ============================================================

ask_tab, about_tab = st.tabs(
    [
        "💬  Ask your PDF",
        "✨  About & Help",
    ]
)


# ============================================================
# TAB 1 — ASK YOUR PDF
# ============================================================

with ask_tab:

    st.markdown(
        """
<div class="hero">

    <div class="hero-badge">
        📚 Intelligent Document Understanding
    </div>

    <h1>
        Ask your PDF anything.
    </h1>

    <p>
        Upload a document and let DocuMind understand
        its text, tables, charts and visuals. Then ask
        questions naturally and get answers grounded
        in the document.
    </p>

</div>
""",
        unsafe_allow_html=True,
    )


    # ========================================================
    # DOCUMENT NOT READY
    # ========================================================

    if not st.session_state.ready:

        st.markdown(
            """
<div class="upload-card">

    <div class="upload-icon">
        📤
    </div>

    <div class="upload-title">
        Drop your PDF here
    </div>

    <div class="upload-text">
        Upload a PDF to create your temporary document
        knowledge base.
    </div>

</div>
""",
            unsafe_allow_html=True,
        )


        uploaded = st.file_uploader(

            "Upload PDF",

            type=["pdf"],

            label_visibility="collapsed",

            help=(
                "Upload a PDF to build a "
                "searchable knowledge base "
                "for this session."
            ),

        )


        if uploaded:

            st.markdown(
                f"""
<div class="document-ready">

    <div class="document-name">
        📄 {uploaded.name}
    </div>

    <div class="document-status">
        {uploaded.size / 1024 / 1024:.2f} MB
        · PDF document
        · Ready to process
    </div>

</div>
""",
                unsafe_allow_html=True,
            )


            if st.button(

                "🚀  Process PDF",

                type="primary",

                use_container_width=True,

            ):

                jina_key = get_jina_key()

                gemini_key = get_gemini_key()

                missing = []

                if not jina_key:

                    missing.append(
                        "JINA_API_KEY"
                    )

                if not gemini_key:

                    missing.append(
                        "GEMINI_API_KEY"
                    )

                if missing:

                    st.error(

                        "Missing API key(s): "
                        + ", ".join(missing)
                        + ". Add them in "
                          "Streamlit Secrets."

                    )

                else:

                    progress = st.progress(
                        0,
                        "Starting…"
                    )

                    try:

                        result = extract_pdf(

                            uploaded.getvalue(),

                            uploaded.name,

                            progress,

                        )

                        st.session_state.ready = True

                        st.session_state.file_name = (
                            uploaded.name
                        )

                        st.session_state.pages = (
                            result["pages"]
                        )

                        st.session_state.chunks = (
                            result["chunks"]
                        )

                        st.session_state.metadata = (
                            result["metadata"]
                        )

                        st.session_state.index = (
                            result["index"]
                        )

                        st.session_state.stats = (
                            result["stats"]
                        )

                        st.session_state.messages = []

                        progress.empty()

                        st.rerun()

                    except Exception as exc:

                        progress.empty()

                        st.error(
                            f"Processing failed: {exc}"
                        )


        else:

            st.markdown(
                """
<div class="card">

    <b>🧠 What DocuMind can understand</b>

    <br><br>

    <div class="pipeline">

        <span class="pipeline-step">
            📄 PDF Text
        </span>

        <span class="pipeline-step">
            📊 Tables
        </span>

        <span class="pipeline-step">
            📈 Charts
        </span>

        <span class="pipeline-step">
            🖼️ Visuals
        </span>

        <span class="pipeline-step">
            🔎 Page-aware Retrieval
        </span>

        <span class="pipeline-step">
            💬 Natural Questions
        </span>

    </div>

</div>
""",
                unsafe_allow_html=True,
            )


    # ========================================================
    # DOCUMENT READY
    # ========================================================

    else:

        stats = st.session_state.stats


        # ----------------------------------------------------
        # DOCUMENT HEADER
        # ----------------------------------------------------

        col_a, col_b = st.columns(
            [5, 1]
        )

        with col_a:

            st.markdown(
                f"""
<div class="document-ready">

    <div class="document-name">
        📄 {st.session_state.file_name}
    </div>

    <div class="document-status">
        ✓ Document processed and ready
    </div>

</div>
""",
                unsafe_allow_html=True,
            )

        with col_b:

            if st.button(
                "🗑️ Clear",
                use_container_width=True
            ):

                clear_document()


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        metrics = [

            (
                c1,
                "📄",
                st.session_state.pages,
                "Pages"
            ),

            (
                c2,
                "🧩",
                stats.get(
                    "chunks",
                    0
                ),
                "Chunks"
            ),

            (
                c3,
                "📊",
                stats.get(
                    "tables",
                    0
                ),
                "Tables"
            ),

            (
                c4,
                "🖼️",
                stats.get(
                    "visuals",
                    0
                ),
                "Visuals"
            ),

        ]


        for col, (
            icon,
            value,
            label
        ) in [

            (
                item[0],
                item[1],
                item[2],
                item[3]
            )

            for item in metrics

        ]:

            with col:

                st.markdown(

                    f"""
<div class="metric-card">

    <div class="metric-icon">
        {icon}
    </div>

    <div class="metric-value">
        {value}
    </div>

    <div class="metric-label">
        {label}
    </div>

</div>
""",

                    unsafe_allow_html=True,

                )


        # ----------------------------------------------------
        # CHAT HEADER
        # ----------------------------------------------------

        st.markdown("")

        st.markdown(
            f"""
<div class="gradient-card">

    <h3>
        💬 Chat with your document
    </h3>

    <p>
        Ask questions about the content, tables,
        figures, numbers, findings or specific pages.
    </p>

</div>
""",
            unsafe_allow_html=True,
        )


        # ----------------------------------------------------
        # SUGGESTED QUESTIONS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-label">'
            'Quick questions'
            '</div>',
            unsafe_allow_html=True,
        )


        q1, q2, q3, q4 = st.columns(4)


        suggestions = [

            (
                q1,
                "📌 Summarize",
                "Summarize the key points of this document."
            ),

            (
                q2,
                "🔍 Key findings",
                "What are the key findings in this document?"
            ),

            (
                q3,
                "📊 Tables",
                "Explain the important tables in this document."
            ),

            (
                q4,
                "📈 Charts",
                "What do the important charts or figures show?"
            ),

        ]


        for col, label, question_text in suggestions:

            with col:

                if st.button(
                    label,
                    use_container_width=True
                ):

                    st.session_state.pending_question = (
                        question_text
                    )

                    st.rerun()


        # ----------------------------------------------------
        # CHAT HISTORY
        # ----------------------------------------------------

        for message in st.session_state.messages:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

                if message.get(
                    "sources"
                ):

                    with st.expander(
                        "📚 View supporting pages"
                    ):

                        for src in message[
                            "sources"
                        ]:

                            st.caption(

                                f"Page {src['page']} "
                                f"· similarity "
                                f"{src['score']:.3f}"

                            )


        # ----------------------------------------------------
        # QUESTION INPUT
        # ----------------------------------------------------

        question = st.chat_input(
            "Ask anything about the document…"
        )


        # Suggested question gets priority
        if (
            not question
            and st.session_state.pending_question
        ):

            question = (
                st.session_state.pending_question
            )

            st.session_state.pending_question = None


        # ----------------------------------------------------
        # PROCESS QUESTION
        # ----------------------------------------------------

        if question:

            st.session_state.messages.append({

                "role":
                    "user",

                "content":
                    question,

            })


            with st.chat_message(
                "user"
            ):

                st.markdown(
                    question
                )


            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "🔎 Searching your document…"
                ):

                    try:

                        results = retrieve(
                            question,
                            top_k=5
                        )

                        answer, model = (
                            generate_answer(
                                question,
                                results
                            )
                        )


                        sources = []


                        for item in results:

                            chunk = item[
                                "chunk"
                            ]

                            page = chunk.get(
                                "page_start"
                            )

                            page_end = chunk.get(
                                "page_end"
                            )


                            page_label = (

                                str(page)

                                if (
                                    page == page_end
                                    or page_end is None
                                )

                                else
                                f"{page}-{page_end}"

                            )


                            sources.append({

                                "page":
                                    page_label,

                                "score":
                                    item["score"],

                            })


                        st.markdown(
                            answer
                        )


                        with st.expander(
                            "📚 View supporting pages"
                        ):

                            for src in sources:

                                st.caption(

                                    f"Page {src['page']} "
                                    f"· similarity "
                                    f"{src['score']:.3f}"

                                )


                        st.session_state.messages.append({

                            "role":
                                "assistant",

                            "content":
                                answer,

                            "sources":
                                sources,

                        })


                    except Exception as exc:

                        error_text = (

                            "Sorry, I couldn't "
                            "answer that question."
                            "\n\n"
                            f"`{exc}`"

                        )

                        st.error(
                            error_text
                        )


                        st.session_state.messages.append({

                            "role":
                                "assistant",

                            "content":
                                error_text,

                        })


# ============================================================
# TAB 2 — ABOUT & HELP
# ============================================================

with about_tab:

    st.markdown(
        """
<div class="hero">

    <div class="hero-badge">
        ✨ About DocuMind
    </div>

    <h1>
        Intelligent document understanding.
    </h1>

    <p>
        DocuMind combines document parsing, embeddings,
        vector retrieval and generative AI to make complex
        PDF documents easier to explore.
    </p>

</div>
""",
        unsafe_allow_html=True,
    )


    # ========================================================
    # WHAT IS DOCUMIND
    # ========================================================

    st.markdown(
        "## 🧠 What is DocuMind?"
    )

    st.write(
        "DocuMind is a Retrieval-Augmented Generation "
        "(RAG) based PDF question-answering assistant. "
        "It allows users to upload a PDF and ask questions "
        "about the information contained in that document."
    )


    # ========================================================
    # FEATURES
    # ========================================================

    st.markdown(
        "## ✨ What it can handle"
    )

    a1, a2, a3 = st.columns(3)


    with a1:

        st.markdown(
            """
<div class="about-card">

    <div class="about-icon">
        📄
    </div>

    <div class="about-title">
        Document Text
    </div>

    <div class="about-text">
        Extracts and understands text while
        preserving document structure and
        page information.
    </div>

</div>
""",
            unsafe_allow_html=True,
        )


    with a2:

        st.markdown(
            """
<div class="about-card">

    <div class="about-icon">
        📊
    </div>

    <div class="about-title">
        Tables & Data
    </div>

    <div class="about-text">
        Extracts structured tables and keeps
        table content intact during the
        retrieval process.
    </div>

</div>
""",
            unsafe_allow_html=True,
        )


    with a3:

        st.markdown(
            """
<div class="about-card">

    <div class="about-icon">
        📈
    </div>

    <div class="about-title">
        Visual Understanding
    </div>

    <div class="about-text">
        Important charts, figures and visuals
        can be converted into searchable
        descriptions.
    </div>

</div>
""",
            unsafe_allow_html=True,
        )


    # ========================================================
    # HOW IT WORKS
    # ========================================================

    st.markdown(
        "## ⚙️ How DocuMind works"
    )


    steps = [

        (
            "1",
            "📤",
            "PDF Upload",
            "The user uploads a PDF for the current application session."
        ),

        (
            "2",
            "🔎",
            "Docling",
            "Docling extracts document text, tables and pictures."
        ),

        (
            "3",
            "🖼️",
            "Visual Understanding",
            "Gemini generates searchable descriptions for meaningful visuals."
        ),

        (
            "4",
            "🧩",
            "Structure-aware Chunking",
            "Text is split into meaningful chunks while tables and visual descriptions are preserved."
        ),

        (
            "5",
            "🔢",
            "Jina Embeddings",
            "Jina Embeddings v4 converts document chunks into vector representations."
        ),

        (
            "6",
            "⚡",
            "FAISS Retrieval",
            "The most relevant document sections are retrieved for each question."
        ),

        (
            "7",
            "🤖",
            "Gemini Answer",
            "Gemini generates an answer using the retrieved document context."
        ),

    ]


    for number, icon, title, description in steps:

        st.markdown(

            f"""
<div class="card">

    <b>
        {number}. {icon} {title}
    </b>

    <br>

    <span style="opacity:.68;">
        {description}
    </span>

</div>
""",

            unsafe_allow_html=True,

        )


    # ========================================================
    # PIPELINE
    # ========================================================

    st.markdown(
        "## 🔗 RAG Pipeline"
    )

    st.markdown(
        """
<div class="card">

    <div class="pipeline">

        <span class="pipeline-step">
            📄 PDF
        </span>

        <span>→</span>

        <span class="pipeline-step">
            🔎 Docling
        </span>

        <span>→</span>

        <span class="pipeline-step">
            🧩 Chunking
        </span>

        <span>→</span>

        <span class="pipeline-step">
            🔢 Jina Embeddings
        </span>

        <span>→</span>

        <span class="pipeline-step">
            ⚡ FAISS
        </span>

        <span>→</span>

        <span class="pipeline-step">
            🤖 Gemini
        </span>

        <span>→</span>

        <span class="pipeline-step">
            💬 Answer
        </span>

    </div>

</div>
""",
        unsafe_allow_html=True,
    )


    # ========================================================
    # HOW TO USE
    # ========================================================

    st.markdown(
        "## 🚀 How to use"
    )


    st.markdown(
        """
<div class="card">

<b>Step 1 — Upload</b><br>
Upload your PDF in the <b>Ask your PDF</b> tab.

<br><br>

<b>Step 2 — Process</b><br>
Click <b>Process PDF</b> and wait while the document is processed.

<br><br>

<b>Step 3 — Ask</b><br>
Enter your question in the chat box.

<br><br>

<b>Step 4 — Verify</b><br>
Use <b>View supporting pages</b> below the answer
to see the pages used for the response.

</div>
""",
        unsafe_allow_html=True,
    )


    # ========================================================
    # QUESTION GUIDANCE
    # ========================================================

    st.markdown(
        "## 💡 Question tips"
    )

    st.info(
        "For precise answers, use terminology from the PDF. "
        "For tables and charts, mention the relevant metric, "
        "category, year or other context when useful."
    )


    tips1, tips2 = st.columns(2)


    with tips1:

        st.markdown(
            """
<div class="card">

<b>Good questions</b>

<br><br>

• What are the key findings?<br>
• What was the revenue in 2024?<br>
• Compare the values shown in the table.<br>
• What does Figure 3 show?<br>
• Which page discusses this topic?

</div>
""",
            unsafe_allow_html=True,
        )


    with tips2:

        st.markdown(
            """
<div class="card">

<b>For complex questions</b>

<br><br>

• Mention the year or period.<br>
• Specify the metric.<br>
• Mention the category.<br>
• Ask all parts of a multi-part question.<br>
• Check the supporting pages after the answer.

</div>
""",
            unsafe_allow_html=True,
        )


    # ========================================================
    # SUPPORT
    # ========================================================

    st.markdown(
        "## 📩 Support & Contact"
    )

    st.markdown(
        """
<div class="gradient-card">

    <h3>
        Need help?
    </h3>

    <p>
        For technical support, bug reports or feedback,
        contact the project owner through the support
        channel configured for this deployment.
    </p>

    <br>

    <b>Support email:</b>
    YOUR_EMAIL@example.com

    <br><br>

    <b>GitHub:</b>
    YOUR_GITHUB_REPOSITORY

</div>
""",
        unsafe_allow_html=True,
    )


    # ========================================================
    # PRIVACY NOTE
    # ========================================================

    st.warning(
        "Uploaded documents are processed for the active "
        "application session. Avoid uploading confidential "
        "information unless your deployment and data-handling "
        "policy allow it."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
    DocuMind · Intelligent PDF Assistant
    · Docling · Jina Embeddings · FAISS · Gemini
</div>
""",
    unsafe_allow_html=True,
)
