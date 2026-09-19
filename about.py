"""DocuMind About / Architecture tab."""

import streamlit as st


def render_about():
    st.title("📘 About DocuMind")
    st.caption("Intelligent Document Assistant — ask questions, inspect evidence, and explore your PDF.")

    left, right = st.columns(2)

    with left:
        st.subheader("What it does")
        st.write(
            "DocuMind turns an uploaded PDF into a searchable knowledge base. "
            "It extracts text and tables with Docling, interprets document pictures "
            "with Gemini, creates structure-aware chunks, embeds them with Jina "
            "Embeddings v4, and retrieves evidence through FAISS."
        )

        st.subheader("Architecture")
        st.markdown(
            """
            **PDF**
            → **Docling ingestion**
            → **Text + Tables + Pictures**
            → **Gemini visual descriptions**
            → **Canonical document**
            → **Structure-aware chunks**
            → **Jina Embeddings v4**
            → **FAISS**
            → **Top-k evidence**
            → **Gemini answer**
            """
        )

    with right:
        st.subheader("How the answer is produced")
        st.markdown(
            """
            1. Your question is embedded as a retrieval query.
            2. FAISS finds the most semantically relevant chunks.
            3. The retrieved evidence is supplied to Gemini.
            4. Gemini is instructed to answer only from that evidence.
            5. DocuMind displays page, table, and figure indicators.
            6. You can expand the complete retrieved chunk content.
            """
        )

        st.subheader("How to use")
        st.markdown(
            """
            - Upload a **PDF** in the **Document Q&A** tab.
            - Click **Process document**.
            - Wait for the live processing stages to finish.
            - Ask questions using the chat box.
            - Use **Exact word check** when you need literal word matching.
            - Open **Retrieved evidence** to inspect the source chunks.
            - Use **Clear document** before starting a different document.
            """
        )

    st.divider()
    st.subheader("Support & help")
    st.info(
        "For deployment or project support, replace the contact details below "
        "with your preferred support channel before publishing."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("App", "DocuMind")
    c2.metric("Mode", "PDF RAG")
    c3.metric("Interface", "Streamlit")

    st.markdown(
        """
        **Contact:** gopinath.sara1n@gmail.com
        **Project documentation:** See `README.md` in the repository.

        **Important:** Never commit API keys to GitHub. Store `JINA_API_KEY`
        and `GEMINI_API_KEY` in Streamlit Secrets.
        """
    )

    with st.expander("Technology stack"):
        st.markdown(
            """
            - Streamlit — application UI and deployment
            - Docling — PDF parsing, tables, and picture extraction
            - Gemini — visual understanding and answer generation
            - Jina Embeddings v4 — retrieval embeddings
            - FAISS — vector similarity search
            - PyPDF — PDF page count
            """
        )
