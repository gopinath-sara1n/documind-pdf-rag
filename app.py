import streamlit as st
from pathlib import Path

from rag_pipeline import (
    extract_pdf,
    retrieve,
    build_context,
    generate_answer,
    clear_backend_cache,
)


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
# CUSTOM CSS
# IMPORTANT:
# Visible UI content is NOT written as raw HTML.
# HTML is used ONLY for CSS.
# ============================================================

st.markdown(
    """
    <style>

    /* -----------------------------
       Global
    ----------------------------- */

    .stApp {
        background: linear-gradient(
            180deg,
            #f8faff 0%,
            #ffffff 38%,
            #f8fafc 100%
        );
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* -----------------------------
       Header
    ----------------------------- */

    .brand-icon {
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 25px;
        box-shadow: 0 8px 22px rgba(79, 70, 229, 0.22);
    }

    .brand-title {
        font-size: 24px;
        font-weight: 800;
        color: #111827;
        line-height: 1.1;
    }

    .brand-subtitle {
        color: #64748b;
        font-size: 13px;
        margin-top: 3px;
    }

    .status-pill {
        background: #eef2ff;
        color: #4f46e5;
        border: 1px solid #c7d2fe;
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
        text-align: center;
    }

    /* -----------------------------
       Hero
    ----------------------------- */

    .hero-title {
        font-size: 46px;
        font-weight: 850;
        line-height: 1.08;
        letter-spacing: -1.8px;
        color: #111827;
        margin-bottom: 12px;
    }

    .hero-description {
        color: #64748b;
        font-size: 17px;
        line-height: 1.7;
        max-width: 760px;
    }

    /* -----------------------------
       Cards
    ----------------------------- */

    .card-title {
        font-size: 18px;
        font-weight: 800;
        color: #111827;
    }

    .card-text {
        color: #64748b;
        line-height: 1.6;
    }

    /* -----------------------------
       Upload area
    ----------------------------- */

    .upload-title {
        font-size: 21px;
        font-weight: 800;
        color: #111827;
        text-align: center;
    }

    .upload-description {
        color: #64748b;
        text-align: center;
        line-height: 1.6;
    }

    /* -----------------------------
       Capability badges
    ----------------------------- */

    .badge {
        display: inline-block;
        padding: 9px 13px;
        border-radius: 999px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #334155;
        font-size: 13px;
        font-weight: 650;
        margin: 3px;
    }

    /* -----------------------------
       Document ready
    ----------------------------- */

    .ready-title {
        font-size: 22px;
        font-weight: 800;
        color: #166534;
    }

    .ready-text {
        color: #475569;
        line-height: 1.6;
    }

    /* -----------------------------
       Chat
    ----------------------------- */

    .chat-header {
        background: linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed
        );
        color: white;
        padding: 20px 24px;
        border-radius: 18px;
        margin-bottom: 18px;
        box-shadow: 0 10px 28px rgba(79, 70, 229, 0.18);
    }

    .chat-header-title {
        font-size: 21px;
        font-weight: 800;
    }

    .chat-header-text {
        font-size: 13px;
        opacity: 0.88;
        margin-top: 4px;
    }

    /* -----------------------------
       Metric cards
    ----------------------------- */

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 14px;
    }

    /* -----------------------------
       Buttons
    ----------------------------- */

    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        border: 1px solid #e2e8f0;
        min-height: 42px;
    }

    /* -----------------------------
       Tabs
    ----------------------------- */

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f1f5f9;
        padding: 5px;
        border-radius: 13px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        padding: 8px 18px;
        font-weight: 700;
    }

    /* -----------------------------
       Footer
    ----------------------------- */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        padding-top: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "document_data" not in st.session_state:
    st.session_state.document_data = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "processing" not in st.session_state:
    st.session_state.processing = False


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns([7, 2])

with header_left:
    col_icon, col_brand = st.columns([0.65, 5])

    with col_icon:
        st.markdown(
            '<div class="brand-icon">📄</div>',
            unsafe_allow_html=True,
        )

    with col_brand:
        st.markdown(
            '<div class="brand-title">DocuMind</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="brand-subtitle">Intelligent PDF Assistant</div>',
            unsafe_allow_html=True,
        )

with header_right:
    st.markdown(
        '<div class="status-pill">✦ AI-Powered RAG</div>',
        unsafe_allow_html=True,
    )


st.write("")


# ============================================================
# TOP NAVIGATION
# ============================================================

tab_ask, tab_about = st.tabs(
    [
        "💬 Ask your PDF",
        "✨ About & Help",
    ]
)


# ============================================================
# ASK TAB
# ============================================================

with tab_ask:

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    st.markdown(
        "### 📚 Intelligent Document Understanding"
    )

    st.markdown(
        '<div class="hero-title">Ask your PDF anything.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-description">
        Upload a document and let DocuMind understand its
        text, tables, charts and visuals. Then ask questions
        naturally and get answers grounded in the document.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.write("")


    # --------------------------------------------------------
    # UPLOAD CARD
    # --------------------------------------------------------

    with st.container(border=True):

        st.markdown(
            '<div class="upload-title">📤 Drop your PDF here</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="upload-description">
            Upload a PDF to create your temporary document
            knowledge base.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        uploaded_file = st.file_uploader(
            "Choose a PDF",
            type=["pdf"],
            accept_multiple_files=False,
            help="Maximum file size: 500 MB",
            label_visibility="collapsed",
        )

        st.caption("📦 Maximum file size: **500 MB** • 📄 PDF only")


    st.write("")


    # --------------------------------------------------------
    # PROCESS DOCUMENT
    # --------------------------------------------------------

    if uploaded_file is not None:

        new_document = (
            st.session_state.document_name
            != uploaded_file.name
        )

        if new_document:

            process_col1, process_col2 = st.columns(
                [5, 1]
            )

            with process_col1:
                st.info(
                    f"📄 **{uploaded_file.name}** is ready to process."
                )

            with process_col2:
                process_clicked = st.button(
                    "🚀 Process PDF",
                    type="primary",
                    use_container_width=True,
                )

            if process_clicked:

                st.session_state.processing = True

                with st.status(
                    "Processing your PDF...",
                    expanded=True,
                ) as status:

                    st.write("📖 Reading PDF structure...")
                    st.write("📊 Extracting text and tables...")
                    st.write("🖼️ Processing visual content...")
                    st.write("🔎 Creating searchable embeddings...")
                    st.write("🧠 Building document knowledge base...")

                    try:

                        document_data = extract_pdf(
                            uploaded_file
                        )

                        st.session_state.document_data = (
                            document_data
                        )

                        st.session_state.document_name = (
                            uploaded_file.name
                        )

                        st.session_state.chat_history = []

                        status.update(
                            label="Document ready!",
                            state="complete",
                            expanded=False,
                        )

                        st.session_state.processing = False
                        st.rerun()

                    except Exception as e:

                        status.update(
                            label="Processing failed",
                            state="error",
                            expanded=True,
                        )

                        st.session_state.processing = False

                        st.error(
                            "The PDF could not be processed."
                        )

                        with st.expander(
                            "Technical details"
                        ):
                            st.exception(e)


    # --------------------------------------------------------
    # CAPABILITIES
    # --------------------------------------------------------

    with st.container(border=True):

        st.markdown(
            '<div class="card-title">🧠 What DocuMind can understand</div>',
            unsafe_allow_html=True,
        )

        st.write("")

        badges = [
            "📄 PDF Text",
            "📊 Tables",
            "📈 Charts",
            "🖼️ Visuals",
            "🔎 Page-aware Retrieval",
            "💬 Natural Questions",
        ]

        badge_html = "".join(
            f'<span class="badge">{badge}</span>'
            for badge in badges
        )

        st.markdown(
            f'<div>{badge_html}</div>',
            unsafe_allow_html=True,
        )


    # ========================================================
    # DOCUMENT READY SECTION
    # ========================================================

    if st.session_state.document_data is not None:

        data = st.session_state.document_data


        st.write("")
        st.write("")


        # ----------------------------------------------------
        # READY CARD
        # ----------------------------------------------------

        with st.container(border=True):

            st.markdown(
                '<div class="ready-title">✅ Document ready</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="ready-text">
                <b>{st.session_state.document_name}</b>
                has been processed and is ready for questions.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")


            # Safely read metrics
            pages = data.get("pages", 0)
            chunks = data.get("chunks", 0)
            tables = data.get("tables", 0)
            visuals = data.get("visuals", 0)


            metric1, metric2, metric3, metric4 = st.columns(4)

            with metric1:
                st.metric(
                    "📄 Pages",
                    pages,
                )

            with metric2:
                st.metric(
                    "🧩 Chunks",
                    chunks,
                )

            with metric3:
                st.metric(
                    "📊 Tables",
                    tables,
                )

            with metric4:
                st.metric(
                    "🖼️ Visuals",
                    visuals,
                )


        st.write("")


        # ----------------------------------------------------
        # CHAT HEADER
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="chat-header">
                <div class="chat-header-title">
                    💬 Ask your document
                </div>
                <div class="chat-header-text">
                    Answers are generated from the retrieved
                    content of your uploaded PDF.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


        # ----------------------------------------------------
        # QUICK QUESTIONS
        # ----------------------------------------------------

        st.markdown("**⚡ Quick questions**")

        q1, q2, q3, q4 = st.columns(4)

        quick_question = None

        with q1:
            if st.button(
                "📌 Summarize",
                use_container_width=True,
            ):
                quick_question = (
                    "Summarize the main points of this document."
                )

        with q2:
            if st.button(
                "🔍 Key findings",
                use_container_width=True,
            ):
                quick_question = (
                    "What are the key findings or important points in this document?"
                )

        with q3:
            if st.button(
                "📊 Tables",
                use_container_width=True,
            ):
                quick_question = (
                    "What important information is contained in the tables?"
                )

        with q4:
            if st.button(
                "📈 Charts",
                use_container_width=True,
            ):
                quick_question = (
                    "Explain the important charts, graphs, or visual data in the document."
                )


        # ----------------------------------------------------
        # CHAT HISTORY
        # ----------------------------------------------------

        for message in st.session_state.chat_history:

            role = message.get("role")
            content = message.get("content")
            pages_used = message.get("pages", [])

            if role == "user":

                with st.chat_message("user"):
                    st.write(content)

            else:

                with st.chat_message("assistant"):

                    st.write(content)

                    if pages_used:

                        unique_pages = sorted(
                            set(pages_used)
                        )

                        page_text = ", ".join(
                            str(p)
                            for p in unique_pages
                        )

                        st.caption(
                            f"📍 Supporting pages: {page_text}"
                        )


        # ----------------------------------------------------
        # QUESTION INPUT
        # ----------------------------------------------------

        user_question = st.chat_input(
            "Ask a question about your PDF..."
        )


        if quick_question is not None:
            user_question = quick_question


        # ----------------------------------------------------
        # ANSWER QUESTION
        # ----------------------------------------------------

        if user_question:

            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": user_question,
                    "pages": [],
                }
            )

            with st.chat_message("user"):
                st.write(user_question)


            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching the document..."
                ):

                    try:

                        retrieved = retrieve(
                            user_question,
                            data,
                            top_k=5,
                        )

                        context = build_context(
                            retrieved
                        )

                        answer = generate_answer(
                            user_question,
                            context,
                        )


                        # Extract page numbers
                        supporting_pages = []

                        for item in retrieved:

                            page = item.get(
                                "page"
                            )

                            if page is not None:
                                supporting_pages.append(
                                    page
                                )


                        st.write(answer)


                        if supporting_pages:

                            unique_pages = sorted(
                                set(
                                    supporting_pages
                                )
                            )

                            page_text = ", ".join(
                                str(p)
                                for p in unique_pages
                            )

                            st.caption(
                                f"📍 Supporting pages: {page_text}"
                            )


                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "pages": supporting_pages,
                            }
                        )


                    except Exception as e:

                        st.error(
                            "I couldn't generate an answer."
                        )

                        with st.expander(
                            "Technical details"
                        ):
                            st.exception(e)


        # ----------------------------------------------------
        # RESET DOCUMENT
        # ----------------------------------------------------

        st.write("")
        st.divider()

        reset_col1, reset_col2 = st.columns(
            [7, 2]
        )

        with reset_col1:
            st.caption(
                "🔒 Your uploaded document is used only for this session."
            )

        with reset_col2:

            if st.button(
                "🗑️ Clear document",
                use_container_width=True,
            ):

                clear_backend_cache()

                st.session_state.document_data = None
                st.session_state.document_name = None
                st.session_state.chat_history = []

                st.rerun()


    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.markdown(
        '<div class="footer">DocuMind • Intelligent PDF Assistant • AI-Powered RAG</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# ABOUT TAB
# ============================================================

with tab_about:

    st.markdown(
        "## ✨ About DocuMind"
    )

    st.markdown(
        """
        **DocuMind** is an Intelligent Document Assistant
        designed to let you interact naturally with complex
        PDF documents.
        """
    )

    st.write("")


    # --------------------------------------------------------
    # WHAT IT DOES
    # --------------------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 🧠 What does DocuMind do?"
        )

        st.write(
            """
            DocuMind processes your PDF, extracts meaningful
            document content, converts that information into
            searchable representations, retrieves the most
            relevant sections for your question, and generates
            an answer grounded in the retrieved document content.
            """
        )


    st.write("")


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.markdown("### 🚀 Key capabilities")

    f1, f2, f3 = st.columns(3)

    with f1:

        with st.container(border=True):

            st.markdown("### 📄 Text")

            st.write(
                "Understands structured PDF text and preserves document context."
            )

    with f2:

        with st.container(border=True):

            st.markdown("### 📊 Tables")

            st.write(
                "Extracts tables and makes their information searchable."
            )

    with f3:

        with st.container(border=True):

            st.markdown("### 🖼️ Visuals")

            st.write(
                "Uses AI-generated descriptions to make visual content searchable."
            )


    st.write("")


    # --------------------------------------------------------
    # RAG PIPELINE
    # --------------------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 🔎 How the RAG pipeline works"
        )

        st.markdown(
            """
            **1. Upload PDF**  
            Your document is temporarily processed.

            **2. Document parsing**  
            Docling extracts text, tables and visual content.

            **3. Visual understanding**  
            Gemini generates descriptions for relevant visual content.

            **4. Chunking**  
            Document content is divided into searchable chunks.

            **5. Embeddings**  
            Jina Embeddings converts document chunks into vectors.

            **6. Vector search**  
            FAISS retrieves the most relevant chunks for your question.

            **7. Answer generation**  
            Gemini generates an answer using the retrieved document context.
            """
        )


    st.write("")


    # --------------------------------------------------------
    # USAGE TIPS
    # --------------------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 💡 Tips for better answers"
        )

        st.markdown(
            """
            - Ask specific questions about the document.
            - Mention a topic, section, table, or page when useful.
            - For numerical information, ask directly about the relevant table.
            - For charts, ask the assistant to explain the trend or comparison.
            - Ask follow-up questions to explore the same document.
            """
        )


    st.write("")


    # --------------------------------------------------------
    # PRIVACY
    # --------------------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 🔒 Privacy"
        )

        st.write(
            """
            Documents uploaded to DocuMind are processed for the
            current application session. Avoid uploading confidential
            or sensitive documents unless you are comfortable with
            the external AI services configured for this application.
            """
        )


    st.write("")


    # --------------------------------------------------------
    # SUPPORT
    # --------------------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 🛠️ Support"
        )

        st.write(
            """
            If you encounter a processing or deployment issue,
            check the Streamlit application logs and verify that
            the required API secrets are correctly configured.
            """
        )


    st.markdown(
        '<div class="footer">DocuMind • Intelligent PDF Assistant</div>',
        unsafe_allow_html=True,
    )
