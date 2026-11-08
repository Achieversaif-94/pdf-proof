import streamlit as st
import os
import shutil
from src.ingest import process_pdf
from src.pipeline import PDFProofPipeline

# --- Page Config ---
st.set_page_config(page_title="PDF Proof", page_icon="🛡️", layout="wide")

# --- Hide Streamlit Header (GitHub icon) ---
hide_header = """
<style>
header[data-testid="stHeader"] {
    display: none;
}
</style>
"""
st.markdown(hide_header, unsafe_allow_html=True)

st.title("🛡️ PDF Proof — Hallucination-Resistant Q&A")
st.markdown("Upload a PDF, ask a question, and get an answer with **page-level citations**. If the AI can't prove it, it won't say it.")

# --- Initialize Session State ---
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# --- Sidebar: Upload PDF ---
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Extracting text, chunking, and embedding..."):
                # 1. Save the uploaded file to the data folder
                os.makedirs("data", exist_ok=True)
                pdf_path = f"data/{uploaded_file.name}"
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 2. Clear the old database (so we don't mix documents)
                if os.path.exists("./chroma_db"):
                    shutil.rmtree("./chroma_db")
                
                # 3. Run ingestion
                process_pdf(pdf_path, collection_name="pdf_docs")
                
                # 4. Initialize the pipeline
                st.session_state.pipeline = PDFProofPipeline()
                st.session_state.pipeline_ready = True
                st.session_state.pdf_name = uploaded_file.name
                st.success("PDF processed successfully!")
                st.rerun()

# --- Main Area ---
if not st.session_state.pipeline_ready:
    st.info("👈 Please upload a PDF in the sidebar to get started.")
else:
    st.header(f"Ask a question about: `{st.session_state.pdf_name}`")
    
    query = st.text_input("Your Question:", placeholder="e.g., What is the BLEU score of the Transformer?")
    
    if st.button("Get Answer", type="primary"):
        if query.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching document and verifying answer..."):
                # Run the pipeline
                result = st.session_state.pipeline.ask(query)
                
                # 1. Display the Answer
                st.subheader("🤖 AI Answer")
                st.write(result["answer"])
                
                # 2. Display Verification Badge
                verif = result["verification"]
                if verif["is_valid"]:
                    st.success(f"✅ **Verified:** {verif['reason']} (Confidence: {verif['confidence']}%)")
                else:
                    st.error(f"🚨 **Verification Failed:** {verif['reason']} (Confidence: {verif['confidence']}%)")
                
                # 3. Display Source Chunks (The "Proof")
                with st.expander("📚 View Source Citations (The Proof)"):
                    for i, chunk in enumerate(result["sources"]):
                        st.markdown(f"**Source {i+1} — Page {chunk['page']}**")
                        st.info(chunk["text"])
