from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from about import render_about
from rag_pipeline import (
    delete_document,
    exact_word_search,
    generate_answer,
    process_document,
    retrieve_chunks,
)

st.set_page_config(
    page_title="DocuMind | Intelligent Document Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Native Streamlit UI only — no custom HTML/CSS.
st.title("📘 DocuMind")
st.caption("Intelligent Document Assistant • PDF → Evidence → Answers")

if "doc" not in st.session_state:
    st.session_state.doc = None
if "work_dir" not in st.session_state:
    st.session_state.work_dir = None
if "chat" not in st.session_state:
    st.session_state.chat = []
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "show_chunks" not in st.session_state:
    st.session_state.show_chunks = False
if "word_results" not in st.session_state:
    st.session_state.word_results = []
if "word_query" not in st.session_state:
    st.session_state.word_query = ""

tab_main, tab_about = st.tabs(["💬 Document Q&A", "ℹ️ About DocuMind"])

with tab_main:
    # ---------------------------------------------------------
    # Header / controls
    # ---------------------------------------------------------
    top1, top2, top3 = st.columns([2.4, 1.1, 1.1])
    with top1:
        st.subheader("1. Upload and process a document")
    with top2:
        if st.session_state.doc:
            st.success("Ready")
    with top3:
        if st.button("🗑️ Clear document", use_container_width=True):
            if st.session_state.work_dir:
                delete_document(st.session_state.work_dir)
            st.session_state.doc = None
            st.session_state.work_dir = None
            st.session_state.chat = []
            st.session_state.last_results = []
            st.session_state.word_results = []
            st.session_state.word_query = ""
            st.session_state.show_chunks = False
            st.rerun()

    uploaded = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        accept_multiple_files=False,
        help="Upload one PDF at a time.",
    )

    process_col, info_col = st.columns([1, 2])
    with process_col:
        process_clicked = st.button(
            "🚀 Process document",
            type="primary",
            use_container_width=True,
            disabled=uploaded is None,
        )
    with info_col:
        if uploaded:
            st.info(f"Selected: **{uploaded.name}** • {uploaded.size / 1024 / 1024:.2f} MB")

    if process_clicked and uploaded is not None:
        # Use a fresh directory for each document.
        digest = hashlib.sha256(uploaded.getvalue()).hexdigest()[:16]
        work_dir = Path(tempfile.gettempdir()) / "documind" / digest
        work_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = work_dir / uploaded.name
        pdf_path.write_bytes(uploaded.getvalue())

        progress = st.progress(0, text="Starting DocuMind...")
        status = st.status("Processing document…", expanded=True)

        def update_progress(value, message):
            progress.progress(value, text=message)
            status.write(f"**{value}%** — {message}")

        try:
            with status:
                result = process_document(pdf_path, work_dir, update_progress)

            progress.progress(100, text="Complete")
            st.session_state.doc = result
            st.session_state.work_dir = str(work_dir)
            st.session_state.chat = []
            st.session_state.last_results = []
            st.session_state.word_results = []
            st.session_state.word_query = ""
            st.session_state.show_chunks = False
            st.success("✅ Document is ready for questions.")
            st.rerun()
        except Exception as exc:
            status.update(label="Processing failed", state="error", expanded=True)
            st.error(str(exc))

    # ---------------------------------------------------------
    # Stats
    # ---------------------------------------------------------
    if st.session_state.doc:
        st.divider()
        doc = st.session_state.doc
        st.subheader("2. Document intelligence")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Pages", doc["pages"])
        m2.metric("Chunks", doc["chunks"])
        m3.metric("Tables", doc["tables"])
        m4.metric("Pictures", doc["pictures"])
        m5.metric("Text elements", doc["text_elements"])

        # -----------------------------------------------------
        # Exact word check
        # -----------------------------------------------------
        st.divider()
        st.subheader("🔎 Exact word check")
        st.caption("Literal word matching across all processed chunks. Matching is case-insensitive.")
        wc1, wc2 = st.columns([4, 1])
        with wc1:
            word = st.text_input(
                "Word",
                value=st.session_state.word_query,
                placeholder="Example: revenue",
                label_visibility="collapsed",
            )
        with wc2:
            check_word = st.button("Check word", use_container_width=True)

        if check_word:
            st.session_state.word_query = word
            if word.strip():
                st.session_state.word_results = exact_word_search(
                    word,
                    st.session_state.work_dir,
                )
            else:
                st.session_state.word_results = []

        if st.session_state.word_query:
            matches = st.session_state.word_results
            if matches:
                st.success(f"Found **{len(matches)}** matching chunk(s) for `{st.session_state.word_query}`.")
                for chunk in matches[:50]:
                    label = (
                        f"{chunk['chunk_id']} • "
                        f"Page {chunk['page_start']}"
                        + (f"–{chunk['page_end']}" if chunk["page_end"] != chunk["page_start"] else "")
                    )
                    with st.expander(label):
                        st.markdown(chunk["content"])
            else:
                st.warning(f"No exact word match found for `{st.session_state.word_query}`.")

        # -----------------------------------------------------
        # Main workspace: PDF + chat
        # -----------------------------------------------------
        st.divider()
        pdf_col, chat_col = st.columns([1, 1.15], gap="large")

        with pdf_col:
            st.subheader("📄 PDF viewer")
            pdf_path = Path(st.session_state.work_dir) / Path(doc["pdf_path"]).name
            if pdf_path.exists():
                try:
                    st.pdf(str(pdf_path), height=650)
                except Exception:
                    st.info("PDF viewer is unavailable in this environment. The uploaded PDF is still processed normally.")
            else:
                st.info("PDF preview is unavailable after the temporary file was removed.")

        with chat_col:
            st.subheader("💬 Ask your document")
            st.caption("Answers are grounded in the retrieved document context.")

            for message in st.session_state.chat:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message.get("meta"):
                        st.caption(message["meta"])

            question = st.chat_input("Ask a question about this PDF…")

            if question:
                st.session_state.chat.append({
                    "role": "user",
                    "content": question,
                })

                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("Retrieving evidence and generating answer…"):
                        results = retrieve_chunks(
                            question,
                            st.session_state.work_dir,
                            top_k=5,
                        )
                        answer, model_used = generate_answer(question, results)

                    st.markdown(answer)

                    source_bits = []
                    for item in results:
                        c = item["chunk"]
                        page = (
                            str(c["page_start"])
                            if c["page_start"] == c["page_end"]
                            else f"{c['page_start']}–{c['page_end']}"
                        )
                        markers = []
                        if c.get("has_table"):
                            markers.append("table")
                        if c.get("has_visual"):
                            markers.append("figure")
                        marker_text = f" • {', '.join(markers)}" if markers else ""
                        source_bits.append(f"p. {page}{marker_text}")

                    meta = (
                        f"Sources: {' | '.join(source_bits)} • "
                        f"Top-k: {len(results)} • Model: {model_used}"
                    )
                    st.caption(meta)

                st.session_state.last_results = results
                st.session_state.chat.append({
                    "role": "assistant",
                    "content": answer,
                    "meta": meta,
                })

                st.rerun()

        # -----------------------------------------------------
        # Retrieved evidence
        # -----------------------------------------------------
        if st.session_state.last_results:
            st.divider()
            ev1, ev2 = st.columns([3, 1])
            with ev1:
                st.subheader("📚 Retrieved evidence")
                st.caption("The five chunks supplied to the answer generator.")
            with ev2:
                if st.button(
                    "📝 Show all chunk content",
                    use_container_width=True,
                ):
                    st.session_state.show_chunks = not st.session_state.show_chunks

            if st.session_state.show_chunks:
                for item in st.session_state.last_results:
                    c = item["chunk"]
                    badges = []
                    if c.get("has_table"):
                        badges.append("TABLE")
                    if c.get("has_visual"):
                        badges.append("FIGURE")
                    badge_text = f" • {' • '.join(badges)}" if badges else ""
                    title = (
                        f"#{item['rank']} {c['chunk_id']} • "
                        f"Page {c['page_start']}–{c['page_end']} • "
                        f"score {item['score']:.4f}{badge_text}"
                    )
                    with st.expander(title):
                        st.markdown(c["content"])
                        st.caption(
                            f"Section: {c.get('section') or '—'} • "
                            f"Elements: {', '.join(c.get('element_types', []))}"
                        )

with tab_about:
    render_about()
