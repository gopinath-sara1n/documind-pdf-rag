import streamlit as st

from rag_pipeline import (
    process_pdf,
    retrieve,
    generate_answer,
    clear_backend_cache,
)

from about import render_about


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DocuMind | Intelligent PDF Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
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
# CSS
# ============================================================

st.markdown(
    """
<style>

/* ---------------------------------------------------------
   Hide default Streamlit elements
--------------------------------------------------------- */

#MainMenu,
footer,
header {
    visibility: hidden;
}


/* ---------------------------------------------------------
   Main container
--------------------------------------------------------- */

.block-container {

    max-width: 1180px;

    padding-top: 1.4rem;

    padding-bottom: 2rem;

}


/* ---------------------------------------------------------
   Brand
--------------------------------------------------------- */

.brand-row {

    display: flex;

    align-items: center;

    justify-content: space-between;

    gap: 20px;

    margin-bottom: 24px;

}


.brand-left {

    display: flex;

    align-items: center;

    gap: 13px;

}


.brand-icon {

    width: 48px;

    height: 48px;

    border-radius: 14px;

    display: flex;

    align-items: center;

    justify-content: center;

    font-size: 25px;

    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed
        );

    box-shadow:
        0 8px 24px
        rgba(79,70,229,.22);

}


.brand-title {

    font-size: 25px;

    font-weight: 800;

    line-height: 1.05;

    color: #111827;

}


.brand-subtitle {

    font-size: 13px;

    color: #64748b;

    margin-top: 4px;

}


.status-pill {

    padding: 9px 15px;

    border-radius: 999px;

    background: #eef2ff;

    border: 1px solid #c7d2fe;

    color: #4338ca;

    font-size: 13px;

    font-weight: 700;

    white-space: nowrap;

}


/* ---------------------------------------------------------
   Hero
--------------------------------------------------------- */

.hero {

    padding: 42px 38px;

    border-radius: 25px;

    background:
        linear-gradient(
            135deg,
            #eef2ff 0%,
            #f5f3ff 48%,
            #ecfeff 100%
        );

    border: 1px solid #e0e7ff;

    margin-bottom: 24px;

}


.hero-badge {

    display: inline-block;

    padding: 8px 13px;

    border-radius: 999px;

    background: white;

    border: 1px solid #ddd6fe;

    color: #5b21b6;

    font-size: 13px;

    font-weight: 700;

    margin-bottom: 16px;

}


.hero h1 {

    margin: 0;

    font-size: 42px;

    line-height: 1.1;

    color: #111827;

    letter-spacing: -1px;

}


.hero p {

    max-width: 760px;

    margin: 14px 0 0;

    color: #475569;

    font-size: 16px;

    line-height: 1.7;

}


/* ---------------------------------------------------------
   Upload card
--------------------------------------------------------- */

.upload-card {

    min-height: 260px;

    padding: 34px 25px 28px;

    border-radius: 24px;

    border: 2px dashed #c7d2fe;

    background:
        linear-gradient(
            180deg,
            #ffffff,
            #f8faff
        );

    text-align: center;

    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: center;

}


.upload-icon {

    width: 64px;

    height: 64px;

    border-radius: 20px;

    display: flex;

    align-items: center;

    justify-content: center;

    font-size: 31px;

    background: #eef2ff;

    margin-bottom: 15px;

}


.upload-title {

    font-size: 22px;

    font-weight: 800;

    color: #111827;

}


.upload-text {

    color: #64748b;

    margin-top: 7px;

    max-width: 520px;

    line-height: 1.55;

}


.file-limit {

    text-align: center;

    color: #64748b;

    font-size: 13px;

    margin: 8px 0 18px;

}


/* ---------------------------------------------------------
   Streamlit uploader
--------------------------------------------------------- */

div[data-testid="stFileUploader"] {

    margin-top: -84px;

    position: relative;

    z-index: 2;

}


div[data-testid="stFileUploaderDropzone"] {

    min-height: 150px;

    border: 0 !important;

    background: transparent !important;

    box-shadow: none !important;

}


div[data-testid="stFileUploaderDropzoneInstructions"] {

    padding-top: 90px;

}


/* ---------------------------------------------------------
   Capabilities
--------------------------------------------------------- */

.capability-card {

    margin-top: 24px;

    padding: 24px;

    border-radius: 22px;

    background: #ffffff;

    border: 1px solid #e5e7eb;

    box-shadow:
        0 8px 30px
        rgba(15,23,42,.05);

}


.capability-title {

    font-size: 17px;

    font-weight: 800;

    color: #111827;

    margin-bottom: 16px;

}


.pipeline {

    display: flex;

    flex-wrap: wrap;

    gap: 10px;

}


.pipeline-step {

    padding: 9px 13px;

    border-radius: 999px;

    background: #f8fafc;

    border: 1px solid #e2e8f0;

    color: #334155;

    font-size: 13px;

    font-weight: 600;

}


/* ---------------------------------------------------------
   Document ready
--------------------------------------------------------- */

.document-ready {

    padding: 22px 24px;

    border-radius: 20px;

    background:
        linear-gradient(
            135deg,
            #f0fdf4,
            #ecfdf5
        );

    border: 1px solid #bbf7d0;

    margin-bottom: 18px;

}


.document-name {

    font-size: 20px;

    font-weight: 800;

    color: #14532d;

}


.document-status {

    color: #166534;

    font-size: 13px;

    margin-top: 4px;

}


/* ---------------------------------------------------------
   Metrics
--------------------------------------------------------- */

.metric-card {

    padding: 19px;

    min-height: 112px;

    border-radius: 18px;

    background: #ffffff;

    border: 1px solid #e5e7eb;

    box-shadow:
        0 5px 22px
        rgba(15,23,42,.045);

}


.metric-icon {

    font-size: 21px;

}


.metric-value {

    font-size: 25px;

    font-weight: 800;

    color: #111827;

    margin-top: 5px;

}


.metric-label {

    color: #64748b;

    font-size: 12px;

}


/* ---------------------------------------------------------
   Chat header
--------------------------------------------------------- */

.gradient-card {

    margin-top: 22px;

    padding: 23px;

    border-radius: 22px;

    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed
        );

    color: white;

    box-shadow:
        0 12px 35px
        rgba(79,70,229,.2);

}


/* ---------------------------------------------------------
   Section label
--------------------------------------------------------- */

.section-label {

    font-size: 13px;

    font-weight: 800;

    letter-spacing: .4px;

    text-transform: uppercase;

    color: #64748b;

    margin: 24px 0 10px;

}


/* ---------------------------------------------------------
   About
--------------------------------------------------------- */

.about-card {

    padding: 24px;

    border-radius: 20px;

    background: #ffffff;

    border: 1px solid #e5e7eb;

    margin-bottom: 15px;

}


.about-icon {

    font-size: 27px;

}


.about-title {

    font-size: 18px;

    font-weight: 800;

    margin-top: 8px;

    color: #111827;

}


.about-text {

    color: #64748b;

    line-height: 1.65;

    margin-top: 6px;

}


/* ---------------------------------------------------------
   Buttons
--------------------------------------------------------- */

div.stButton > button {

    border-radius: 12px;

    font-weight: 700;

    min-height: 42px;

}


/* ---------------------------------------------------------
   Footer
--------------------------------------------------------- */

.footer {

    text-align: center;

    color: #94a3b8;

    font-size: 12px;

    padding: 28px 0 5px;

}


/* ---------------------------------------------------------
   Mobile
--------------------------------------------------------- */

@media (max-width: 700px) {

    .hero {

        padding: 30px 22px;

    }

    .hero h1 {

        font-size: 32px;

    }

    .brand-title {

        font-size: 21px;

    }

    .status-pill {

        font-size: 11px;

        padding: 8px 10px;

    }

}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HEADER
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
    unsafe_allow_html=True
)


# ============================================================
# TABS
# ============================================================

ask_tab, about_tab = st.tabs(
    [
        "💬 Ask your PDF",
        "✨ About & Help"
    ]
)


# ============================================================
# ASK TAB
# ============================================================

with ask_tab:

    # ========================================================
    # BEFORE PDF PROCESSING
    # ========================================================

    if not st.session_state.ready:

        # ----------------------------------------------------
        # Hero
        # ----------------------------------------------------

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
                    its text, tables, charts and visuals.
                    Then ask questions naturally and get answers
                    grounded in the document.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # Upload visual card
        # ----------------------------------------------------

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
                    Upload a PDF to create your temporary
                    document knowledge base.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # File limit
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="file-limit">
                📦 Maximum file size:
                <b>500 MB</b>
                &nbsp;•&nbsp;
                📄 PDF only
            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # Actual Streamlit uploader
        # ----------------------------------------------------

        uploaded = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            label_visibility="collapsed",
            key="pdf_uploader"
        )


        # ----------------------------------------------------
        # Selected file
        # ----------------------------------------------------

        if uploaded:

            file_size_mb = (
                uploaded.size
                /
                (1024 * 1024)
            )

            st.markdown(
                f"""
                <div class="document-ready">

                    <div class="document-name">
                        📄 {uploaded.name}
                    </div>

                    <div class="document-status">
                        Ready to create a temporary
                        knowledge base
                        · {file_size_mb:.1f} MB
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # Process button
            # ------------------------------------------------

            if st.button(
                "🚀 Process PDF",
                type="primary",
                use_container_width=True
            ):

                with st.status(
                    "Building your document knowledge base...",
                    expanded=True
                ) as status:

                    try:

                        progress = st.empty()

                        progress.write(
                            "📄 Extracting text, tables and visuals..."
                        )

                        result = process_pdf(
                            uploaded
                        )

                        progress.write(
                            "🧠 Creating embeddings and vector index..."
                        )

                        # Store result
                        st.session_state.chunks = (
                            result["chunks"]
                        )

                        st.session_state.metadata = (
                            result["metadata"]
                        )

                        st.session_state.index = (
                            result["index"]
                        )

                        st.session_state.pages = (
                            result["pages"]
                        )

                        st.session_state.stats = (
                            result["stats"]
                        )

                        st.session_state.file_name = (
                            uploaded.name
                        )

                        st.session_state.messages = []

                        st.session_state.pending_question = None

                        st.session_state.ready = True

                        status.update(
                            label="✅ Document is ready!",
                            state="complete",
                            expanded=False
                        )

                        st.rerun()

                    except Exception as exc:

                        status.update(
                            label="❌ Processing failed",
                            state="error",
                            expanded=True
                        )

                        st.error(
                            f"{type(exc).__name__}: {exc}"
                        )


        # ----------------------------------------------------
        # Capabilities
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="capability-card">

                <div class="capability-title">
                    🧠 What DocuMind can understand
                </div>

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
            unsafe_allow_html=True
        )


    # ========================================================
    # AFTER PDF PROCESSING
    # ========================================================

    else:

        stats = (
            st.session_state.stats
            or {}
        )


        # ----------------------------------------------------
        # Document header
        # ----------------------------------------------------

        document_col, clear_col = st.columns(
            [5, 1]
        )

        with document_col:

            st.markdown(
                f"""
                <div class="document-ready">

                    <div class="document-name">
                        📄 {st.session_state.file_name}
                    </div>

                    <div class="document-status">
                        ● Document ready for questions
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with clear_col:

            if st.button(
                "🗑️ Clear",
                use_container_width=True
            ):

                clear_backend_cache()

                for key, value in DEFAULT_STATE.items():

                    st.session_state[key] = value

                st.rerun()


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = [

            (
                "📄",
                st.session_state.pages,
                "Pages"
            ),

            (
                "🧩",
                len(st.session_state.chunks),
                "Chunks"
            ),

            (
                "📊",
                stats.get(
                    "tables",
                    0
                ),
                "Tables"
            ),

            (
                "🖼️",
                stats.get(
                    "visuals",
                    0
                ),
                "Visuals"
            ),

        ]

        metric_columns = st.columns(4)

        for col, (
            icon,
            value,
            label
        ) in zip(
            metric_columns,
            metrics
        ):

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
                    unsafe_allow_html=True
                )


        # ----------------------------------------------------
        # Chat header
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="gradient-card">

                <b>
                    💬 Ask your PDF
                </b>

                <br>

                <span style="opacity:.9">
                    Questions are answered using
                    the retrieved document context.
                </span>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # Quick questions
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-label">'
            'Quick questions'
            '</div>',
            unsafe_allow_html=True
        )

        quick_questions = [

            (
                "📌 Summarize",
                "Summarize the document."
            ),

            (
                "🔍 Key findings",
                "What are the key findings in the document?"
            ),

            (
                "📊 Tables",
                "What are the most important tables and what do they show?"
            ),

            (
                "📈 Charts",
                "What do the important charts or figures show?"
            ),

        ]

        quick_columns = st.columns(4)

        for col, (
            label,
            question
        ) in zip(
            quick_columns,
            quick_questions
        ):

            with col:

                if st.button(
                    label,
                    use_container_width=True
                ):

                    st.session_state.pending_question = (
                        question
                    )

                    st.rerun()


        # ----------------------------------------------------
        # Existing conversation
        # ----------------------------------------------------

        if st.session_state.messages:

            st.markdown(
                '<div class="section-label">'
                'Conversation'
                '</div>',
                unsafe_allow_html=True
            )


        for message in st.session_state.messages:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

                if message.get(
                    "pages"
                ):

                    st.caption(
                        "📄 Supporting pages: "
                        +
                        ", ".join(
                            map(
                                str,
                                message["pages"]
                            )
                        )
                    )


        # ----------------------------------------------------
        # Chat input
        # ----------------------------------------------------

        question = st.chat_input(
            "Ask a question about your PDF..."
        )


        # ----------------------------------------------------
        # Quick-question handling
        # ----------------------------------------------------

        if st.session_state.pending_question:

            question = (
                st.session_state.pending_question
            )

            st.session_state.pending_question = None


        # ----------------------------------------------------
        # Process question
        # ----------------------------------------------------

        if question:

            # -----------------------------------------------
            # User message
            # -----------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.chat_message(
                "user"
            ):

                st.markdown(
                    question
                )


            # -----------------------------------------------
            # Assistant
            # -----------------------------------------------

            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "Searching the document..."
                ):

                    try:

                        results = retrieve(
                            question,
                            st.session_state.index,
                            st.session_state.chunks,
                            st.session_state.metadata,
                            top_k=5
                        )


                        answer = generate_answer(
                            question,
                            results
                        )


                        # -----------------------------------
                        # Supporting pages
                        # -----------------------------------

                        pages = sorted(
                            {
                                int(
                                    result["metadata"]["page"]
                                )

                                for result in results

                                if result["metadata"].get(
                                    "page"
                                ) is not None
                            }
                        )


                        st.markdown(
                            answer
                        )


                        if pages:

                            st.caption(
                                "📄 Supporting pages: "
                                +
                                ", ".join(
                                    map(
                                        str,
                                        pages
                                    )
                                )
                            )


                        # -----------------------------------
                        # Save assistant message
                        # -----------------------------------

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "pages": pages
                            }
                        )


                    except Exception as exc:

                        error_text = (
                            f"❌ Unable to answer: "
                            f"{type(exc).__name__}: {exc}"
                        )

                        st.error(
                            error_text
                        )

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": error_text
                            }
                        )


# ============================================================
# ABOUT TAB
# ============================================================

with about_tab:

    render_about()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        DocuMind · Temporary document knowledge base ·
        Answers are grounded in retrieved PDF content

    </div>
    """,
    unsafe_allow_html=True
)
