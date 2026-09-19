"""
DocuMind - RAG pipeline

Architecture adapted from the uploaded notebook:
PDF -> Docling -> text/tables/pictures
     -> Gemini visual descriptions
     -> canonical document
     -> structure-aware chunks
     -> Jina Embeddings v4
     -> FAISS
Question -> Jina query embedding -> FAISS top-k -> Gemini answer

The Colab/Google Drive-specific parts of the notebook are replaced with
per-document temporary/local storage so the same architecture can run in
Streamlit Community Cloud.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional

import faiss
import numpy as np
import pandas as pd
import requests
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from google import genai


ProgressFn = Optional[Callable[[int, str], None]]

JINA_MODEL = "jina-embeddings-v4"
JINA_DIMENSIONS = 2048
JINA_ENDPOINT = "https://api.jina.ai/v1/embeddings"
JINA_BATCH_SIZE = 100
JINA_MIN_INTERVAL = 0.20
JINA_MAX_RETRIES = 6

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
]

TARGET_CHARS = 1800
MAX_CHARS = 3200
MIN_STANDALONE_CHARS = 150


def _emit(cb: ProgressFn, value: int, message: str) -> None:
    if cb:
        cb(max(0, min(100, int(value))), message)


def _secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st
        value = str(st.secrets.get(name, "")).strip()
    except Exception:
        value = ""
    return value


def _clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _attr(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _get_item_text(item) -> str:
    for key in ("text", "orig", "content"):
        value = _attr(item, key, None)
        if isinstance(value, str) and value.strip():
            return _clean_text(value)
    return ""


def _page_no(item):
    prov = _attr(item, "prov", []) or []
    if prov:
        return _attr(prov[0], "page_no", None)
    return None


def _table_to_text(table) -> str:
    try:
        dataframe = table.export_to_dataframe()
        if dataframe is not None:
            dataframe = dataframe.fillna("")
            lines = []
            columns = [_clean_text(x) for x in dataframe.columns]
            if any(columns):
                lines.append(" | ".join(columns))
            for _, row in dataframe.iterrows():
                values = [_clean_text(v) for v in row.tolist()]
                if any(values):
                    lines.append(" | ".join(v if v else "-" for v in values))
            result = "\n".join(lines).strip()
            if result:
                return result
    except Exception:
        pass
    return _get_item_text(table)


def _resolve_ref(ref, maps):
    if isinstance(ref, dict):
        ref = ref.get("$ref")
    if not isinstance(ref, str):
        return None
    for item_type, mapping in maps.items():
        if ref in mapping:
            return item_type, mapping[ref]
    return None


def _extract_canonical(docling_data, gemini_results):
    texts = {x.get("self_ref"): x for x in docling_data.get("texts", []) if x.get("self_ref")}
    tables = {x.get("self_ref"): x for x in docling_data.get("tables", []) if x.get("self_ref")}
    pictures = {x.get("self_ref"): x for x in docling_data.get("pictures", []) if x.get("self_ref")}
    groups = {x.get("self_ref"): x for x in docling_data.get("groups", []) if x.get("self_ref")}

    picture_ref_to_id = {}
    for idx, picture in enumerate(docling_data.get("pictures", []), 1):
        ref = picture.get("self_ref")
        if ref:
            picture_ref_to_id[ref] = f"picture_{idx:03d}"

    gemini_by_picture = {
        x.get("picture_id"): x
        for x in gemini_results
        if x.get("status") == "success"
    }

    maps = {"text": texts, "table": tables, "picture": pictures, "group": groups}
    elements = []

    for position, child in enumerate(docling_data.get("body", {}).get("children", [])):
        resolved = _resolve_ref(child, maps)
        if not resolved:
            continue

        item_type, item = resolved
        page = _page_no(item)

        if item_type == "text":
            text = _get_item_text(item)
            if not text:
                continue
            elements.append({
                "element_id": f"element_{len(elements)+1:06d}",
                "type": "text",
                "position": position,
                "page": page,
                "text": text,
                "label": item.get("label", "text"),
                "source": {"docling_ref": item.get("self_ref")},
            })

        elif item_type == "table":
            table_text = _table_to_text(item)
            if table_text:
                elements.append({
                    "element_id": f"element_{len(elements)+1:06d}",
                    "type": "table",
                    "position": position,
                    "page": page,
                    "text": table_text,
                    "content": table_text,
                    "source": {"docling_ref": item.get("self_ref")},
                })

        elif item_type == "picture":
            picture_ref = item.get("self_ref")
            picture_id = picture_ref_to_id.get(picture_ref)
            visual = gemini_by_picture.get(picture_id)
            if not visual:
                continue
            description = _clean_text(visual.get("description", ""))
            if not description:
                continue
            elements.append({
                "element_id": f"element_{len(elements)+1:06d}",
                "type": "visual",
                "position": position,
                "page": page,
                "picture_id": picture_id,
                "image_path": visual.get("image_path"),
                "visual_description": description,
                "text": description,
                "content": description,
                "source": {
                    "docling_ref": picture_ref,
                    "gemini_model": visual.get("model"),
                },
            })

    chunking_elements = [
        e for e in elements
        if not (
            e["type"] == "text"
            and e.get("label", "text") in {"page_header", "page_footer"}
        )
    ]

    return {
        "schema_version": "1.0",
        "document_name": docling_data.get("name"),
        "statistics": {
            "total_elements": len(elements),
            "text_elements": sum(e["type"] == "text" for e in elements),
            "table_elements": sum(e["type"] == "table" for e in elements),
            "visual_elements": sum(e["type"] == "visual" for e in elements),
        },
        "elements": elements,
    }, chunking_elements


def _structure_aware_chunks(elements):
    chunks = []
    current_parts = []
    current_ids = []
    current_types = []
    current_pages = []
    current_section = ""

    def current_text():
        return "\n\n".join(x for x in current_parts if x).strip()

    def flush():
        nonlocal current_parts, current_ids, current_types, current_pages
        text = current_text()
        if text:
            chunks.append({
                "chunk_id": f"chunk_{len(chunks):06d}",
                "page_start": min(current_pages) if current_pages else None,
                "page_end": max(current_pages) if current_pages else None,
                "section": current_section,
                "content": text,
                "element_ids": current_ids.copy(),
                "element_types": current_types.copy(),
                "has_table": "table" in current_types,
                "has_visual": "visual" in current_types,
            })
        current_parts = []
        current_ids = []
        current_types = []
        current_pages = []

    def add_piece(element, piece):
        nonlocal current_parts, current_ids, current_types, current_pages
        current_parts.append(piece)
        current_ids.append(element["element_id"])
        current_types.append(element["type"])
        if element.get("page") is not None:
            current_pages.append(element["page"])

    for element in elements:
        etype = element["type"]
        content = _clean_text(element.get("content") or element.get("text"))
        if not content:
            continue

        if etype in {"table", "visual"}:
            if current_text():
                flush()
            add_piece(element, content)
            flush()
            continue

        label = element.get("label", "")
        if label in {"section_header", "title", "heading"}:
            current_section = content
            if current_text():
                flush()
            add_piece(element, content)
            continue

        # Sentence-aware text chunking, preserving the notebook's
        # target/max character strategy.
        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", content)
            if s.strip()
        ]
        if not sentences:
            sentences = [content]

        for sentence in sentences:
            candidate = (
                f"{current_text()}\n\n{sentence}".strip()
                if current_text() else sentence
            )
            if current_text() and len(candidate) > MAX_CHARS:
                flush()
                candidate = sentence

            add_piece(element, sentence)

            if len(current_text()) >= TARGET_CHARS:
                flush()

    if current_text():
        flush()

    # Merge tiny text-only chunks where possible.
    merged = []
    for chunk in chunks:
        if (
            merged
            and len(chunk["content"]) < MIN_STANDALONE_CHARS
            and not chunk["has_table"]
            and not chunk["has_visual"]
            and len(merged[-1]["content"]) + len(chunk["content"]) + 2 <= MAX_CHARS
        ):
            merged[-1]["content"] += "\n\n" + chunk["content"]
            merged[-1]["page_end"] = chunk["page_end"] or merged[-1]["page_end"]
            merged[-1]["element_ids"].extend(chunk["element_ids"])
            merged[-1]["element_types"].extend(chunk["element_types"])
            merged[-1]["has_table"] |= chunk["has_table"]
            merged[-1]["has_visual"] |= chunk["has_visual"]
        else:
            merged.append(chunk)

    for i, chunk in enumerate(merged):
        chunk["chunk_id"] = f"chunk_{i:06d}"

    return merged


def _retryable(text: str) -> bool:
    upper = text.upper()
    return any(x in upper for x in (
        "429", "RATE LIMIT", "TOO MANY REQUESTS",
        "500", "502", "503", "504", "UNAVAILABLE",
        "INTERNAL", "TIMEOUT", "DEADLINE", "RESOURCE_EXHAUSTED"
    ))


def _jina_request(input_texts, task, api_key, timeout=180):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": JINA_MODEL,
        "input": input_texts,
        "task": task,
        "embedding_type": "float",
        "dimensions": JINA_DIMENSIONS,
        "truncate": False,
    }
    last = None
    for attempt in range(1, JINA_MAX_RETRIES + 1):
        try:
            response = requests.post(
                JINA_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Jina HTTP {response.status_code}: {response.text[:500]}"
                )
            body = response.json()
            data = sorted(body.get("data", []), key=lambda x: int(x["index"]))
            if len(data) != len(input_texts):
                raise RuntimeError("Jina returned an unexpected number of embeddings.")
            vectors = [np.asarray(x["embedding"], dtype="float32") for x in data]
            for vector in vectors:
                if vector.shape[0] != JINA_DIMENSIONS or not np.isfinite(vector).all():
                    raise RuntimeError("Invalid Jina embedding returned.")
            return vectors, body.get("usage", {})
        except Exception as exc:
            last = exc
            if not _retryable(str(exc)) or attempt == JINA_MAX_RETRIES:
                raise
            time.sleep(min(2 ** (attempt - 1) + random.uniform(0, 1), 30))
    raise last


def _embed_chunks(chunks, api_key, progress_cb):
    texts = [
        f"title: {c.get('section') or 'none'} | text: {c['content']}"
        for c in chunks
    ]
    vectors = []
    total_batches = max(1, int(np.ceil(len(texts) / JINA_BATCH_SIZE)))
    for start in range(0, len(texts), JINA_BATCH_SIZE):
        batch = texts[start:start + JINA_BATCH_SIZE]
        batch_vectors, _ = _jina_request(batch, "retrieval.passage", api_key)
        vectors.extend(batch_vectors)
        done = min(start + len(batch), len(texts))
        percent = 60 + int(20 * done / len(texts))
        _emit(progress_cb, percent, f"Embedding chunks with Jina: {done}/{len(texts)}")
        if start + JINA_BATCH_SIZE < len(texts):
            time.sleep(JINA_MIN_INTERVAL)
    matrix = np.vstack(vectors).astype("float32")
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    return matrix


def _gemini_describe_images(image_paths, api_key, progress_cb):
    client = genai.Client(api_key=api_key)
    results = []
    total = len(image_paths)

    for i, (picture_id, image_path, page_no) in enumerate(image_paths, 1):
        last = None
        description = ""
        model_used = None

        for model_name in GEMINI_MODELS:
            try:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()

                # google-genai accepts inline image bytes through Part.
                from google.genai import types
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/png",
                        ),
                        (
                            "Describe this document figure/image for retrieval. "
                            "Capture visible text, labels, chart/table meaning, "
                            "relationships, entities, and important numerical values. "
                            "Do not invent details. Return concise factual prose."
                        ),
                    ],
                )
                description = _clean_text(response.text or "")
                if description:
                    model_used = model_name
                    break
            except Exception as exc:
                last = exc
                if not _retryable(str(exc)):
                    break
                time.sleep(min(2 ** (i - 1), 10))

        results.append({
            "picture_id": picture_id,
            "page": page_no,
            "image_path": str(image_path),
            "description": description,
            "model": model_used,
            "status": "success" if description else "error",
            "error": str(last) if last and not description else None,
        })
        _emit(
            progress_cb,
            35 + int(20 * i / max(1, total)),
            f"Generating visual descriptions: {i}/{total}",
        )
    return results


def process_document(pdf_path: str | Path, work_dir: str | Path,
                     progress_cb: ProgressFn = None) -> Dict:
    pdf_path = Path(pdf_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    image_dir = work_dir / "images"
    image_dir.mkdir(exist_ok=True)

    jina_key = _secret("JINA_API_KEY")
    gemini_key = _secret("GEMINI_API_KEY")
    if not jina_key:
        raise ValueError("JINA_API_KEY is missing. Add it to Streamlit secrets.")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY is missing. Add it to Streamlit secrets.")

    _emit(progress_cb, 3, "Opening PDF and preparing Docling...")
    options = PdfPipelineOptions()
    options.do_table_structure = True
    options.generate_picture_images = True
    options.do_picture_description = False
    options.do_ocr = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options)
        }
    )

    _emit(progress_cb, 8, "Docling is extracting text, tables, and pictures...")
    result = converter.convert(str(pdf_path))
    doc = result.document
    docling_data = doc.export_to_dict()

    with open(work_dir / "docling_document.json", "w", encoding="utf-8") as f:
        json.dump(docling_data, f, ensure_ascii=False, indent=2)

    picture_manifest = []
    pictures = getattr(doc, "pictures", []) or []
    for idx, picture in enumerate(pictures, 1):
        picture_id = f"picture_{idx:03d}"
        image_path = image_dir / f"{picture_id}.png"
        try:
            image = picture.get_image(doc)
            if image is not None:
                image.save(image_path)
                picture_manifest.append({
                    "picture_id": picture_id,
                    "image_path": str(image_path),
                    "page": _page_no(picture),
                })
        except Exception:
            continue

    _emit(progress_cb, 30, f"Docling complete. Found {len(pictures)} pictures.")

    gemini_results = _gemini_describe_images(
        [(x["picture_id"], x["image_path"], x["page"]) for x in picture_manifest],
        gemini_key,
        progress_cb,
    )

    canonical, elements = _extract_canonical(docling_data, gemini_results)
    with open(work_dir / "canonical_document.json", "w", encoding="utf-8") as f:
        json.dump(canonical, f, ensure_ascii=False, indent=2)

    _emit(progress_cb, 58, "Building structure-aware chunks...")
    chunks = _structure_aware_chunks(elements)
    with open(work_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    _emit(progress_cb, 60, f"Created {len(chunks)} chunks. Starting Jina embeddings...")
    embeddings = _embed_chunks(chunks, jina_key, progress_cb)

    _emit(progress_cb, 82, "Building FAISS vector index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(work_dir / "faiss.index"))

    metadata = []
    for i, chunk in enumerate(chunks):
        metadata.append({
            "faiss_index": i,
            "chunk_id": chunk["chunk_id"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "section": chunk["section"],
            "element_ids": chunk["element_ids"],
            "element_types": chunk["element_types"],
            "has_table": chunk["has_table"],
            "has_visual": chunk["has_visual"],
        })

    with open(work_dir / "embedding_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    np.save(work_dir / "embeddings.npy", embeddings)

    _emit(progress_cb, 100, "Document processing completed.")
    return {
        "pdf_path": str(pdf_path),
        "work_dir": str(work_dir),
        "pages": _pdf_pages(pdf_path),
        "chunks": len(chunks),
        "tables": canonical["statistics"]["table_elements"],
        "pictures": canonical["statistics"]["visual_elements"],
        "text_elements": canonical["statistics"]["text_elements"],
        "chunk_data": chunks,
    }


def _pdf_pages(pdf_path: Path) -> int:
    from pypdf import PdfReader
    return len(PdfReader(str(pdf_path)).pages)


def load_index(work_dir: str | Path):
    work_dir = Path(work_dir)
    with open(work_dir / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(work_dir / "embedding_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    index = faiss.read_index(str(work_dir / "faiss.index"))
    return chunks, metadata, index


def retrieve_chunks(question: str, work_dir: str | Path, top_k: int = 5):
    jina_key = _secret("JINA_API_KEY")
    if not jina_key:
        raise ValueError("JINA_API_KEY is missing.")
    chunks, metadata, index = load_index(work_dir)
    vectors, _ = _jina_request([question], "retrieval.query", jina_key, timeout=60)
    query = vectors[0]
    query = query / max(np.linalg.norm(query), 1e-12)
    scores, indices = index.search(query.reshape(1, -1), top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        if idx < 0:
            continue
        results.append({
            "rank": rank,
            "score": float(score),
            "chunk": chunks[idx],
            "metadata": metadata[idx],
        })
    return results


def exact_word_search(word: str, work_dir: str | Path):
    chunks, _, _ = load_index(work_dir)
    word = word.strip()
    if not word:
        return []
    pattern = re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE)
    matches = []
    for chunk in chunks:
        if pattern.search(chunk.get("content", "")):
            matches.append(chunk)
    return matches


def _answer_retryable(text: str) -> bool:
    return _retryable(text)


def generate_answer(question: str, results: List[Dict]):
    gemini_key = _secret("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY is missing.")

    client = genai.Client(api_key=gemini_key)
    context_parts = []
    for item in results:
        c = item["chunk"]
        context_parts.append(
            f"SOURCE {item['rank']}\n"
            f"Similarity score: {item['score']:.4f}\n"
            f"Chunk ID: {c['chunk_id']}\n"
            f"Page: {c['page_start']} - {c['page_end']}\n"
            f"Section: {c.get('section')}\n"
            f"Has table: {c.get('has_table')}\n"
            f"Has visual: {c.get('has_visual')}\n\n"
            f"CONTENT:\n{c['content']}"
        )
    context = "\n\n" + ("\n\n" + "=" * 60 + "\n\n").join(context_parts)

    prompt = f"""
You are DocuMind, a document question-answering assistant.

Use ONLY the retrieved document context below.
Do not use outside knowledge and do not invent information.
Compare all retrieved sources before answering.
Pay attention to dates, units, categories, metrics, geography,
and time periods. Preserve numerical values.
If the context is insufficient, say so clearly.
Answer every part of a multi-part question.
Use concise Markdown. Do not mention FAISS, embeddings, chunks,
or retrieval unless the user asks about the system.

After the answer, add a short "Sources" section with the relevant
page numbers and identify a table or figure when the source metadata
indicates one.

RETRIEVED DOCUMENT CONTEXT
==========================
{context}

USER QUESTION
=============
{question}

FINAL ANSWER
============
"""

    last_error = None
    for model_name in GEMINI_MODELS:
        for attempt in range(1, 5):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if response.text:
                    return response.text.strip(), model_name
                raise RuntimeError("Gemini returned an empty answer.")
            except Exception as exc:
                last_error = exc
                if not _answer_retryable(str(exc)) or attempt == 4:
                    break
                time.sleep(min(3 * (2 ** (attempt - 1)) + random.uniform(0, 2), 30))
    raise RuntimeError("All configured Gemini models failed.") from last_error


def delete_document(work_dir: str | Path):
    path = Path(work_dir)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
