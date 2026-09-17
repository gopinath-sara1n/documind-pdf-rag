
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
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# CUSTOM UI
# ------------------------------------------------------------

st.markdown("""
<style>
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #111827 0%, #1f2937 55%, #374151 100%);
        color: white;
        margin-bottom: 1.25rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.35rem;
        letter-spacing: -0.04em;
    }

    .hero p {
        margin: 0.5rem 0 0 0;
        color: #d1d5db;
        font-size: 1.03rem;
    }

    .card {
        padding: 1rem 1.15rem;
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 16px;
        margin-bottom: .8rem;
        background: rgba(128,128,128,.035);
    }

    .metric-card {
        padding: .9rem 1rem;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,.20);
        text-align: center;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
    }

    .metric-label {
        font-size: .78rem;
        opacity: .7;
    }

    .source-pill {
        display: inline-block;
        padding: .25rem .55rem;
        border-radius: 999px;
        background: rgba(99,102,241,.12);
        font-size: .75rem;
        margin-right: .3rem;
    }

    .stButton > button {
        border-radius: 11px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


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
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CONFIG
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
# DOCELING / TEXT HELPERS
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
        for key in ("text", "orig", "content"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return clean_text(value)

    return ""


def get_page_number(item):
    try:
        prov = getattr(item, "prov", [])
        if prov:
            page_no = getattr(prov[0], "page_no", None)
            if page_no is None and isinstance(prov[0], dict):
                page_no = prov[0].get("page_no")
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
            columns = [clean_text(x) for x in df.columns]
            if any(columns):
                lines.append(" | ".join(columns))

            for _, row in df.iterrows():
                values = [clean_text(v) for v in row.tolist()]
                if any(values):
                    lines.append(" | ".join(v if v else "-" for v in values))

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

    paragraphs = re.split(r"\n\s*\n", text)
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
    pipeline_options.do_ocr = False

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )


def extract_pdf(pdf_bytes, file_name, progress):
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / file_name
        pdf_path.write_bytes(pdf_bytes)

        progress.progress(0.12, "Reading PDF with Docling…")
        result = get_converter().convert(str(pdf_path))
        doc = result.document

        elements = []
        text_count = table_count = visual_count = 0

        # We preserve the document order from Docling.
        for position, (item, level) in enumerate(doc.iterate_items()):
            page = get_page_number(item)
            item_name = type(item).__name__.lower()

            if "table" in item_name:
                content = table_to_text(item)
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

            if "picture" in item_name:
                # Visual understanding is handled separately below.
                continue

            content = get_item_text(item)
            if not content:
                continue

            label = getattr(item, "label", "text")
            elements.append({
                "type": "text",
                "page": page,
                "position": position,
                "content": content,
                "label": str(label),
            })
            text_count += 1

        progress.progress(0.35, "Extracting text and tables…")

        # Visual descriptions are generated from Docling's extracted images.
        pictures = list(getattr(doc, "pictures", []))
        visual_candidates = []

        for idx, picture in enumerate(pictures, start=1):
            try:
                image = picture.get_image(doc)
                if image is None:
                    continue

                width, height = image.size
                if width < 200 or height < 80:
                    continue

                page = get_page_number(picture)
                visual_candidates.append((idx, page, image))
            except Exception:
                continue

        visual_total = len(visual_candidates)

        if visual_total:
            client = genai.Client(api_key=get_gemini_key())
            for n, (idx, page, image) in enumerate(visual_candidates, start=1):
                description = describe_visual(
                    client, image, page, GEMINI_MODELS
                )
                if description:
                    elements.append({
                        "type": "visual",
                        "page": page,
                        "position": 10_000_000 + idx,
                        "content": description,
                        "label": "visual",
                    })
                    visual_count += 1

                progress.progress(
                    0.35 + (0.25 * n / max(visual_total, 1)),
                    f"Understanding visuals ({n}/{visual_total})…",
                )

        elements.sort(key=lambda x: x["position"])

        progress.progress(0.62, "Building structure-aware chunks…")
        chunks = make_chunks(elements)

        progress.progress(0.66, "Creating Jina embeddings…")
        embeddings = embed_documents(
            [format_document_for_embedding(c) for c in chunks],
            progress,
        )

        index = faiss.IndexFlatIP(JINA_DIMENSIONS)
        index.add(embeddings)

        metadata = [
            {
                "chunk_id": c["chunk_id"],
                "page_start": c.get("page_start"),
                "page_end": c.get("page_end"),
                "section": c.get("section", ""),
                "element_types": c.get("element_types", []),
                "has_table": c.get("has_table", False),
                "has_visual": c.get("has_visual", False),
            }
            for c in chunks
        ]

        progress.progress(1.0, "Document is ready.")

        return {
            "chunks": chunks,
            "metadata": metadata,
            "index": index,
            "pages": len(getattr(doc, "pages", {}) or {}),
            "stats": {
                "text": text_count,
                "tables": table_count,
                "visuals": visual_count,
                "chunks": len(chunks),
            },
        }


# ============================================================
# GEMINI VISUAL UNDERSTANDING
# ============================================================

def classify_gemini_error(exc):
    text = str(exc).upper()

    if any(x in text for x in ["401", "403", "UNAUTHENTICATED", "PERMISSION_DENIED"]):
        return "auth"

    if any(x in text for x in ["429", "RESOURCE_EXHAUSTED", "RATE LIMIT", "QUOTA"]):
        return "quota"

    if any(x in text for x in ["503", "UNAVAILABLE", "500", "502", "504", "INTERNAL"]):
        return "temporary"

    return "permanent"


def describe_visual(client, image, page, models):
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
    image.save(buf, format="PNG")
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
                error_type = classify_gemini_error(exc)
                if error_type in {"auth", "permanent"}:
                    break

                if attempt < 2:
                    time.sleep(min(2 ** attempt + random.random(), 8))

    return ""


# ============================================================
# STAGE 2 — STRUCTURE-AWARE CHUNKING
# ============================================================

def make_chunks(elements, target_chars=1800, max_chars=3200):
    chunks = []
    current_parts = []
    current_ids = []
    current_types = []
    current_pages = []
    current_section = ""

    def current_text():
        return "\n\n".join(current_parts).strip()

    def flush():
        nonlocal current_parts, current_ids, current_types, current_pages

        text = current_text()
        if not text:
            current_parts = []
            current_ids = []
            current_types = []
            current_pages = []
            return

        chunks.append({
            "chunk_id": f"chunk_{len(chunks):06d}",
            "page_start": min(current_pages) if current_pages else None,
            "page_end": max(current_pages) if current_pages else None,
            "section": current_section,
            "content": text,
            "element_ids": list(current_ids),
            "element_types": list(current_types),
            "has_table": "table" in current_types,
            "has_visual": "visual" in current_types,
        })

        current_parts = []
        current_ids = []
        current_types = []
        current_pages = []

    def add(content, element):
        content = clean_text(content)
        if not content:
            return
        current_parts.append(content)
        current_ids.append(
            f"element_{element.get('position', len(current_ids)):06d}"
        )
        current_types.append(element["type"])
        if element.get("page") is not None:
            current_pages.append(element["page"])

    for element in elements:
        kind = element["type"]
        label = element.get("label", "")
        content = element["content"]

        if kind == "text" and label == "section_header":
            flush()
            current_section = content
            add(content, element)
            continue

        # Tables and visuals remain intact.
        if kind in {"table", "visual"}:
            if current_text():
                candidate = current_text() + "\n\n" + content
                if len(candidate) > target_chars:
                    flush()

            add(content, element)
            continue

        for sentence in split_sentences(content):
            candidate = (
                current_text() + "\n\n" + sentence
                if current_text()
                else sentence
            )

            if len(candidate) <= target_chars:
                add(sentence, element)
                continue

            flush()

            if len(sentence) <= max_chars:
                add(sentence, element)
            else:
                words = sentence.split()
                buffer = []
                for word in words:
                    test = " ".join(buffer + [word])
                    if len(test) <= max_chars:
                        buffer.append(word)
                    else:
                        if buffer:
                            add(" ".join(buffer), element)
                            flush()
                        buffer = [word]
                if buffer:
                    add(" ".join(buffer), element)

    flush()
    return chunks


# ============================================================
# STAGE 3 — JINA EMBEDDING
# ============================================================

def jina_request(inputs, task):
    key = get_jina_key()
    if not key:
        raise RuntimeError(
            "JINA_API_KEY is missing. Add it to Streamlit Secrets."
        )

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": JINA_MODEL,
        "input": inputs,
        "task": task,
        "embedding_type": "float",
        "dimensions": JINA_DIMENSIONS,
        "truncate": False,
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
                    f"Jina HTTP {response.status_code}: "
                    f"{response.text[:400]}"
                )

            data = response.json().get("data", [])
            data = sorted(data, key=lambda x: x["index"])

            vectors = np.asarray(
                [x["embedding"] for x in data],
                dtype="float32",
            )

            if len(vectors) != len(inputs):
                raise RuntimeError("Jina returned an unexpected number of embeddings.")

            vectors /= np.maximum(
                np.linalg.norm(vectors, axis=1, keepdims=True),
                1e-12,
            )
            return vectors

        except Exception as exc:
            last_error = exc
            if attempt == 5:
                break
            time.sleep(min(2 ** attempt + random.random(), 20))

    raise last_error


def embed_documents(texts, progress):
    all_vectors = []

    total = len(texts)
    for start in range(0, total, JINA_BATCH_SIZE):
        batch = texts[start:start + JINA_BATCH_SIZE]
        vectors = jina_request(batch, "retrieval.passage")
        all_vectors.append(vectors)

        done = min(start + len(batch), total)
        progress.progress(
            0.66 + 0.25 * done / max(total, 1),
            f"Embedding chunks ({done}/{total})…",
        )

    return np.vstack(all_vectors).astype("float32")


def embed_query(question):
    return jina_request([question], "retrieval.query")


def format_document_for_embedding(chunk):
    title = chunk.get("section") or "none"
    return f"title: {title} | text: {chunk['content']}"


# ============================================================
# STAGE 4 — RETRIEVAL + ANSWER
# ============================================================

def retrieve(question, top_k=5):
    q = embed_query(question)
    scores, indices = st.session_state.index.search(q, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue
        results.append({
            "rank": rank,
            "score": float(score),
            "chunk": st.session_state.chunks[idx],
            "metadata": st.session_state.metadata[idx],
        })
    return results


def build_context(results):
    parts = []
    for item in results:
        chunk = item["chunk"]
        page = chunk.get("page_start")
        page_end = chunk.get("page_end")
        page_text = (
            f"page {page}"
            if page == page_end or page_end is None
            else f"pages {page}-{page_end}"
        )

        parts.append(
            f"[Source {item['rank']} | {page_text}]\n"
            f"{chunk['content']}"
        )

    return "\n\n".join(parts)


def generate_answer(question, results):
    key = get_gemini_key()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to Streamlit Secrets."
        )

    client = genai.Client(api_key=key)
    context = build_context(results)

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
                return response.text.strip(), model
        except Exception as exc:
            last_error = exc
            time.sleep(1)

    raise RuntimeError(
        f"Gemini could not generate an answer: {last_error}"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 📄 DocuMind")
    st.caption("Intelligent PDF Assistant")

    st.divider()

    if st.session_state.ready:
        st.success("Document ready")
        st.caption(st.session_state.file_name)

        stats = st.session_state.stats
        st.metric("Pages", st.session_state.pages)
        st.metric("Chunks", stats.get("chunks", 0))

        if st.button("🗑️ Clear document", use_container_width=True):
            for key, value in DEFAULT_STATE.items():
                st.session_state[key] = value
            st.rerun()

    st.divider()
    st.caption("Pipeline")
    st.caption("Docling → Structure-aware chunks → Jina → FAISS → Gemini")


# ============================================================
# PAGE 1 — CHAT
# ============================================================

st.markdown("""
<div class="hero">
    <h1>📄 Ask your PDF</h1>
    <p>Upload a document, let DocuMind understand its text, tables and visuals, then ask questions in natural language.</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.ready:
    st.markdown("### Start with a PDF")

    uploaded = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
        help="Upload a PDF to build a searchable knowledge base for this session.",
    )

    if uploaded:
        st.info(
            f"**{uploaded.name}** · {uploaded.size / 1024 / 1024:.2f} MB"
        )

        if st.button(
            "🚀 Process PDF",
            type="primary",
            use_container_width=True,
        ):
            jina_key = get_jina_key()
            gemini_key = get_gemini_key()

            missing = []
            if not jina_key:
                missing.append("JINA_API_KEY")
            if not gemini_key:
                missing.append("GEMINI_API_KEY")

            if missing:
                st.error(
                    "Missing API key(s): "
                    + ", ".join(missing)
                    + ". Add them in Streamlit Secrets."
                )
            else:
                progress = st.progress(0, "Starting…")
                status = st.empty()

                try:
                    result = extract_pdf(
                        uploaded.getvalue(),
                        uploaded.name,
                        progress,
                    )

                    st.session_state.ready = True
                    st.session_state.file_name = uploaded.name
                    st.session_state.pages = result["pages"]
                    st.session_state.chunks = result["chunks"]
                    st.session_state.metadata = result["metadata"]
                    st.session_state.index = result["index"]
                    st.session_state.stats = result["stats"]
                    st.session_state.messages = []

                    progress.empty()
                    status.empty()
                    st.rerun()

                except Exception as exc:
                    progress.empty()
                    st.error(f"Processing failed: {exc}")

    else:
        st.markdown("""
        <div class="card">
        <b>What this app can understand</b><br><br>
        • Normal PDF text<br>
        • Tables and structured data<br>
        • Charts, figures and images through visual descriptions<br>
        • Page-aware retrieval<br>
        • Natural-language questions and answers
        </div>
        """, unsafe_allow_html=True)

else:
    stats = st.session_state.stats

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("📄", st.session_state.pages, "Pages"),
        ("🧩", stats.get("chunks", 0), "Chunks"),
        ("📊", stats.get("tables", 0), "Tables"),
        ("🖼️", stats.get("visuals", 0), "Visuals"),
    ]

    for col, (icon, value, label) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div>{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown(f"### 💬 Ask questions about **{st.session_state.file_name}**")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("View supporting pages"):
                    for src in message["sources"]:
                        st.caption(
                            f"Page {src['page']} · similarity {src['score']:.3f}"
                        )

    question = st.chat_input(
        "Ask anything about the document…"
    )

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the document…"):
                try:
                    results = retrieve(question, top_k=5)
                    answer, model = generate_answer(question, results)

                    sources = []
                    for item in results:
                        chunk = item["chunk"]
                        page = chunk.get("page_start")
                        page_end = chunk.get("page_end")
                        page_label = (
                            str(page)
                            if page == page_end or page_end is None
                            else f"{page}-{page_end}"
                        )
                        sources.append({
                            "page": page_label,
                            "score": item["score"],
                        })

                    st.markdown(answer)

                    with st.expander("View supporting pages"):
                        for src in sources:
                            st.caption(
                                f"Page {src['page']} · similarity {src['score']:.3f}"
                            )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                except Exception as exc:
                    error_text = f"Sorry, I couldn't answer that question.\n\n`{exc}`"
                    st.error(error_text)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_text,
                    })
