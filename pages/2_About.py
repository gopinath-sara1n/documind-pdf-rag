
import streamlit as st

st.set_page_config(
    page_title="About — DocuMind",
    page_icon="ℹ️",
    layout="wide",
)

st.markdown("""
<style>
.block-container {
    max-width: 1100px;
    padding-top: 2rem;
}
.about-hero {
    padding: 1.6rem;
    border-radius: 22px;
    background: linear-gradient(135deg, #111827, #374151);
    color: white;
    margin-bottom: 1.3rem;
}
.card {
    padding: 1.15rem;
    border-radius: 16px;
    border: 1px solid rgba(128,128,128,.22);
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="about-hero">
    <h1>ℹ️ About DocuMind</h1>
    <p>An intelligent Retrieval-Augmented Generation assistant for complex PDF documents.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("## What is DocuMind?")

st.write(
    "DocuMind lets users upload a PDF and ask questions about its contents. "
    "The system is designed to preserve document structure and make text, "
    "tables and important visuals searchable."
)

st.markdown("## How it works")

steps = [
    ("1", "PDF Upload", "The uploaded PDF is processed during the current session."),
    ("2", "Docling", "Docling extracts document text, tables and pictures."),
    ("3", "Visual Understanding", "Gemini creates searchable descriptions for meaningful visuals."),
    ("4", "Structure-aware Chunking", "Text is split by sections while tables and visual descriptions are kept intact."),
    ("5", "Jina Embeddings", "Document chunks are converted into vector representations using Jina Embeddings v4."),
    ("6", "FAISS Retrieval", "The most relevant document sections are retrieved for each question."),
    ("7", "Gemini Answer", "Gemini generates an answer strictly from the retrieved document context."),
]

for number, title, description in steps:
    st.markdown(
        f"""
        <div class="card">
            <b>{number}. {title}</b><br>
            <span>{description}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("## How to use")

st.markdown("""
1. Open **Ask your PDF** from the sidebar.
2. Upload a PDF.
3. Click **Process PDF** and wait for processing to finish.
4. Type your question in the chat box.
5. Check the **supporting pages** shown below each answer.
""")

st.markdown("## Guidance")

st.info(
    "For precise answers, ask questions using the terminology used in the PDF. "
    "For tables and charts, include the metric, category, year or page context "
    "when relevant."
)

st.markdown("## Support & Contact")

st.markdown("""
<div class="card">
<b>Need help?</b><br><br>
For technical support, bug reports or feedback, contact the project owner through the support channel configured for this deployment.<br><br>
<b>Support email:</b> YOUR_EMAIL@example.com<br>
<b>GitHub:</b> YOUR_GITHUB_REPOSITORY
</div>
""", unsafe_allow_html=True)

st.caption(
    "Note: Uploaded documents are processed for the active application session. "
    "Avoid uploading confidential information unless your deployment and data-handling policy allow it."
)
