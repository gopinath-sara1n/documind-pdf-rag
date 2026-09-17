import streamlit as st


def render_about():

    # ========================================================
    # HERO
    # ========================================================

    st.markdown(
        """
        <div class="hero">

            <div class="hero-badge">
                ✨ About DocuMind
            </div>

            <h1>
                Understand documents,
                not just text.
            </h1>

            <p>
                DocuMind is an intelligent PDF assistant designed
                to retrieve relevant information from complex
                documents and answer questions using
                document-grounded context.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # WHAT DOCUMIND DOES
    # ========================================================

    st.markdown(
        '<div class="section-label">What DocuMind does</div>',
        unsafe_allow_html=True
    )

    cards = [

        (
            "📄",
            "PDF Text",
            "Extracts document text while keeping page "
            "information available for retrieval."
        ),

        (
            "📊",
            "Tables",
            "Preserves tables as searchable document "
            "content for table-based questions."
        ),

        (
            "📈",
            "Charts & Figures",
            "Uses visual descriptions as additional "
            "searchable context for figures and charts."
        ),

        (
            "🔎",
            "Page-aware Retrieval",
            "Retrieves relevant chunks while retaining "
            "their source-page metadata."
        ),

        (
            "💬",
            "Natural Questions",
            "Ask questions in normal language instead "
            "of manually searching through the PDF."
        ),

        (
            "🧠",
            "Grounded Answers",
            "The answer generation step is instructed "
            "to use retrieved document context."
        ),

    ]

    columns = st.columns(2)

    for index, (
        icon,
        title,
        description
    ) in enumerate(cards):

        with columns[index % 2]:

            st.markdown(
                f"""
                <div class="about-card">

                    <div class="about-icon">
                        {icon}
                    </div>

                    <div class="about-title">
                        {title}
                    </div>

                    <div class="about-text">
                        {description}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # HOW IT WORKS
    # ========================================================

    st.markdown(
        '<div class="section-label">How it works</div>',
        unsafe_allow_html=True
    )

    steps = [

        (
            "01",
            "Upload",
            "Upload a PDF to create a temporary "
            "document knowledge base."
        ),

        (
            "02",
            "Extract",
            "Docling processes document text, "
            "tables and pictures."
        ),

        (
            "03",
            "Structure",
            "Content is organized into structure-aware "
            "chunks with page metadata."
        ),

        (
            "04",
            "Embed",
            "Jina Embeddings converts document chunks "
            "into vector representations."
        ),

        (
            "05",
            "Retrieve",
            "FAISS finds the most relevant chunks "
            "for your question."
        ),

        (
            "06",
            "Answer",
            "Gemini generates a response using "
            "the retrieved document context."
        ),

    ]

    for number, title, description in steps:

        st.markdown(
            f"""
            <div class="about-card">

                <div style="
                    display:flex;
                    gap:15px;
                    align-items:flex-start;
                ">

                    <div style="
                        min-width:42px;
                        height:42px;
                        border-radius:13px;
                        background:#eef2ff;
                        color:#4f46e5;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-weight:800;
                    ">
                        {number}
                    </div>

                    <div>

                        <div
                            class="about-title"
                            style="margin-top:0;"
                        >
                            {title}
                        </div>

                        <div class="about-text">
                            {description}
                        </div>

                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # RAG PIPELINE
    # ========================================================

    st.markdown(
        '<div class="section-label">RAG pipeline</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="about-card">

            <div class="pipeline">

                <span class="pipeline-step">
                    📄 PDF
                </span>

                <span class="pipeline-step">
                    →
                </span>

                <span class="pipeline-step">
                    🧩 Docling
                </span>

                <span class="pipeline-step">
                    →
                </span>

                <span class="pipeline-step">
                    ✂️ Chunks
                </span>

                <span class="pipeline-step">
                    →
                </span>

                <span class="pipeline-step">
                    🧠 Jina Embeddings
                </span>

                <span class="pipeline-step">
                    →
                </span>

                <span class="pipeline-step">
                    🔎 FAISS
                </span>

                <span class="pipeline-step">
                    →
                </span>

                <span class="pipeline-step">
                    ✨ Gemini
                </span>

                <span class="pipeline-step">
                    →
                </span>

                <span class="pipeline-step">
                    💬 Answer
                </span>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # QUESTION TIPS
    # ========================================================

    st.markdown(
        '<div class="section-label">Tips for better questions</div>',
        unsafe_allow_html=True
    )

    tips = [

        "Ask specific questions when you need a precise fact.",

        "Mention a section, topic, year, table or figure "
        "when the document contains many similar items.",

        "For comparisons, clearly name the items you want "
        "to compare.",

        "For tables and charts, ask about values, trends "
        "or relationships.",

        "If the retrieved context does not contain the answer, "
        "DocuMind will indicate that there is not enough "
        "information in the retrieved document context.",

    ]

    for tip in tips:

        st.markdown(
            f"- {tip}"
        )


    # ========================================================
    # PRIVACY
    # ========================================================

    st.markdown(
        '<div class="section-label">'
        'Privacy & document handling'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="about-card">

            <div class="about-icon">
                🔐
            </div>

            <div class="about-title">
                Temporary document knowledge base
            </div>

            <div class="about-text">

                Documents uploaded through the application
                are processed to build the active session's
                retrieval index.

                Do not upload confidential or sensitive
                documents unless you understand the deployment,
                storage and API policies of the services used
                by your deployment.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )
