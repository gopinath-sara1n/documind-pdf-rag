import streamlit as st

def render_about():
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">✨ About DocuMind</div>
        <h1>Understand documents, not just text.</h1>
        <p>
            DocuMind is an intelligent PDF assistant designed to retrieve
            relevant information from complex documents and answer questions
            using document-grounded context.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">What DocuMind does</div>', unsafe_allow_html=True)

    cards = [
        ("📄", "PDF Text", "Extracts document text while keeping page information available for retrieval."),
        ("📊", "Tables", "Preserves tables as structured document content so table-based questions can be answered."),
        ("📈", "Charts & Figures", "Uses visual descriptions as additional searchable context for figures and charts."),
        ("🔎", "Page-aware Retrieval", "Retrieves the most relevant chunks and keeps their source-page metadata."),
        ("💬", "Natural Questions", "Ask questions in normal language instead of searching manually through the PDF."),
        ("🧠", "Grounded Answers", "The answer-generation step is instructed to use the retrieved document context."),
    ]

    cols = st.columns(2)
    for i, (icon, title, text) in enumerate(cards):
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="about-card">
                    <div class="about-icon">{icon}</div>
                    <div class="about-title">{title}</div>
                    <div class="about-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-label">How it works</div>', unsafe_allow_html=True)

    steps = [
        ("01", "Upload", "Upload a PDF to create a temporary document knowledge base."),
        ("02", "Extract", "Docling processes document text, tables and pictures."),
        ("03", "Structure", "Content is organized into structure-aware chunks with page metadata."),
        ("04", "Embed", "Document chunks are converted into vector embeddings."),
        ("05", "Retrieve", "FAISS finds the most relevant chunks for your question."),
        ("06", "Answer", "Gemini generates a response using the retrieved document context."),
    ]

    for number, title, text in steps:
        st.markdown(
            f"""
            <div class="about-card">
                <div style="display:flex;gap:15px;align-items:flex-start;">
                    <div style="
                        min-width:42px;height:42px;border-radius:13px;
                        background:#eef2ff;color:#4f46e5;
                        display:flex;align-items:center;justify-content:center;
                        font-weight:800;">
                        {number}
                    </div>
                    <div>
                        <div class="about-title" style="margin-top:0">{title}</div>
                        <div class="about-text">{text}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">RAG pipeline</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card">
        <div class="pipeline">
            <span class="pipeline-step">📄 PDF</span>
            <span class="pipeline-step">→</span>
            <span class="pipeline-step">🧩 Docling</span>
            <span class="pipeline-step">→</span>
            <span class="pipeline-step">✂️ Chunks</span>
            <span class="pipeline-step">→</span>
            <span class="pipeline-step">🧠 Jina Embeddings</span>
            <span class="pipeline-step">→</span>
            <span class="pipeline-step">🔎 FAISS</span>
            <span class="pipeline-step">→</span>
            <span class="pipeline-step">✨ Gemini</span>
            <span class="pipeline-step">→</span>
            <span class="pipeline-step">💬 Answer</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Tips for better questions</div>', unsafe_allow_html=True)

    tips = [
        "Ask specific questions when you need a precise fact.",
        "Mention a section, topic, year or table when the PDF contains many similar items.",
        "For comparisons, clearly name the two or more things you want compared.",
        "For tables and charts, ask what the values, trends or relationships mean.",
        "If the document does not contain the answer, DocuMind may indicate that the retrieved context does not provide enough information.",
    ]

    for tip in tips:
        st.markdown(f"- {tip}")

    st.markdown('<div class="section-label">Privacy & document handling</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card">
        <div class="about-icon">🔐</div>
        <div class="about-title">Temporary document knowledge base</div>
        <div class="about-text">
            Documents uploaded through the app are processed to build the
            active session's retrieval index. Do not upload confidential or
            sensitive documents unless you understand the deployment,
            storage and API policies of the services used by your deployment.
        </div>
    </div>
    """, unsafe_allow_html=True)
