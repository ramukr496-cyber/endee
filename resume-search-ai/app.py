"""
AI Resume Semantic Search System
================================
A Streamlit application that uses Endee vector database for semantic search
over resumes. Upload PDF resumes, convert them to embeddings, and search
using natural language queries.

Author: Your Name
Tech Stack: Python, Streamlit, Sentence Transformers, Endee
"""

import streamlit as st
import os
import uuid
import json
from pathlib import Path

# ------------------------------------------------------------------
# Import our custom modules (we wrote these in separate files)
# ------------------------------------------------------------------
from pdf_parser import extract_text_from_pdf
from embeddings import get_embedding_model, generate_embedding
from endee_client import (
    create_resume_index,
    upsert_resume,
    search_resumes,
    get_index_stats,
    delete_resume,
    list_all_resumes,
)

# ------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS for better styling
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background: #f8f9fa;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .similarity-badge {
        background: #4CAF50;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    .upload-section {
        background: #f0f7ff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px dashed #4a90d9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------
# Session state lets us keep data across Streamlit reruns
if "resumes_uploaded" not in st.session_state:
    st.session_state.resumes_uploaded = 0
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False


# ------------------------------------------------------------------
# Load the Embedding Model (cached so it loads only once)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI embedding model...")
def load_model():
    """Load the sentence transformer model once and cache it."""
    return get_embedding_model()


# ------------------------------------------------------------------
# Sidebar: Configuration & Index Management
# ------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/search-in-list.png",
        width=80,
    )
    st.title("⚙️ Configuration")

    # Endee Server URL
    endee_url = st.text_input(
        "Endee Server URL",
        value="http://localhost:8080/api/v1",
        help="The base URL of your running Endee vector database server.",
    )

    # Auth Token (optional)
    auth_token = st.text_input(
        "Auth Token (optional)",
        value="",
        type="password",
        help="If you started Endee with NDD_AUTH_TOKEN, enter it here.",
    )

    st.divider()

    # Index Management Section
    st.subheader("📦 Index Management")

    if st.button("🔄 Create / Reset Index", use_container_width=True):
        with st.spinner("Creating index in Endee..."):
            success, msg = create_resume_index(endee_url, auth_token)
            if success:
                st.success(msg)
                st.session_state.index_ready = True
            else:
                st.error(msg)

    if st.button("📊 Show Index Stats", use_container_width=True):
        stats = get_index_stats(endee_url, auth_token)
        if stats:
            st.json(stats)
        else:
            st.warning("Could not fetch index stats. Is Endee running?")

    st.divider()

    # Info Section
    st.subheader("ℹ️ About")
    st.markdown(
        """
        **AI Resume Search** uses:
        - **Endee** — vector database
        - **Sentence Transformers** — embeddings
        - **Streamlit** — web UI

        Resumes are converted to vector
        embeddings and stored in Endee
        for fast semantic similarity search.
        """
    )


# ------------------------------------------------------------------
# Main Content Area
# ------------------------------------------------------------------
st.markdown('<div class="main-header">🔍 AI Resume Semantic Search</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload resumes, search with natural language — powered by Endee vector database</div>',
    unsafe_allow_html=True,
)

# Load the embedding model
model = load_model()
st.session_state.model_loaded = True

# ------------------------------------------------------------------
# Tab Layout: Upload | Search | Browse
# ------------------------------------------------------------------
tab_upload, tab_search, tab_browse = st.tabs(["📤 Upload Resumes", "🔍 Search", "📋 Browse All"])

# =========================
# TAB 1: Upload Resumes
# =========================
with tab_upload:
    st.markdown("### Upload PDF Resumes")
    st.markdown("Upload one or more PDF resumes. They will be parsed, converted to embeddings, and stored in Endee.")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload resume PDFs to index them for semantic search.",
    )

    if uploaded_files:
        if st.button("🚀 Process & Index Resumes", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            total = len(uploaded_files)
            success_count = 0

            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name} ({i+1}/{total})...")
                progress_bar.progress((i) / total)

                try:
                    # Step 1: Extract text from the PDF
                    text = extract_text_from_pdf(uploaded_file)

                    if not text or len(text.strip()) < 50:
                        st.warning(f"⚠️ {uploaded_file.name}: Too little text extracted. Skipping.")
                        continue

                    # Step 2: Generate embedding from the extracted text
                    embedding = generate_embedding(model, text)

                    # Step 3: Create a unique ID and metadata
                    resume_id = f"resume_{uuid.uuid4().hex[:12]}"
                    metadata = {
                        "filename": uploaded_file.name,
                        "text_preview": text[:500],  # First 500 chars as preview
                        "char_count": str(len(text)),
                    }

                    # Step 4: Store in Endee
                    ok, msg = upsert_resume(
                        base_url=endee_url,
                        auth_token=auth_token,
                        resume_id=resume_id,
                        embedding=embedding,
                        metadata=metadata,
                        full_text=text,
                    )

                    if ok:
                        success_count += 1
                        st.success(f"✅ {uploaded_file.name} indexed successfully!")
                    else:
                        st.error(f"❌ {uploaded_file.name}: {msg}")

                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")

            progress_bar.progress(1.0)
            status_text.text(f"Done! {success_count}/{total} resumes indexed.")
            st.session_state.resumes_uploaded += success_count
            st.balloons()


# =========================
# TAB 2: Semantic Search
# =========================
with tab_search:
    st.markdown("### Search Resumes with Natural Language")
    st.markdown("Type a query describing the candidate you're looking for. The system will find the most relevant resumes.")

    # Example queries for the user
    st.markdown("**Example queries:**")
    example_cols = st.columns(3)
    example_queries = [
        "Python developer with Django and machine learning experience",
        "Data scientist skilled in NLP and deep learning",
        "Frontend developer with React and TypeScript",
    ]

    # Let users click example queries
    selected_example = None
    for idx, col in enumerate(example_cols):
        with col:
            if st.button(f"💡 {example_queries[idx]}", use_container_width=True):
                selected_example = example_queries[idx]

    # Search input
    search_query = st.text_input(
        "🔍 Enter your search query:",
        value=selected_example if selected_example else "",
        placeholder="e.g., Python developer with Django and machine learning experience",
    )

    # Number of results
    top_k = st.slider("Number of results to return:", min_value=1, max_value=20, value=5)

    if search_query:
        if st.button("🔎 Search", type="primary", use_container_width=True) or selected_example:
            with st.spinner("Searching..."):
                # Step 1: Convert the query to an embedding
                query_embedding = generate_embedding(model, search_query)

                # Step 2: Search Endee for similar vectors
                results = search_resumes(
                    base_url=endee_url,
                    auth_token=auth_token,
                    query_embedding=query_embedding,
                    top_k=top_k,
                )

                if results is None:
                    st.error("❌ Search failed. Make sure Endee is running and the index exists.")
                elif len(results) == 0:
                    st.info("No results found. Try uploading some resumes first!")
                else:
                    st.markdown(f"### Found {len(results)} matching resumes:")
                    st.divider()

                    for rank, result in enumerate(results, 1):
                        similarity = result.get("similarity", 0)
                        meta = result.get("meta", {})
                        filename = meta.get("filename", "Unknown")
                        preview = meta.get("text_preview", "No preview available.")
                        char_count = meta.get("char_count", "N/A")

                        # Determine match quality color
                        if similarity >= 0.7:
                            color = "#4CAF50"  # Green = great match
                            label = "Excellent Match"
                        elif similarity >= 0.5:
                            color = "#FF9800"  # Orange = good match
                            label = "Good Match"
                        else:
                            color = "#f44336"  # Red = weak match
                            label = "Partial Match"

                        # Display result card
                        st.markdown(
                            f"""
                            <div style="background:#f8f9fa; border-left:4px solid {color};
                                        padding:1rem; margin:0.5rem 0; border-radius:0 8px 8px 0;">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <h4 style="margin:0;">#{rank} — 📄 {filename}</h4>
                                    <span style="background:{color}; color:white; padding:4px 12px;
                                                 border-radius:20px; font-weight:600;">
                                        {label} — {similarity:.1%}
                                    </span>
                                </div>
                                <p style="color:#555; margin-top:0.5rem; font-size:0.9rem;">
                                    Characters: {char_count}
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Show expandable text preview
                        with st.expander(f"📝 View Resume Preview — {filename}"):
                            st.text(preview)

                    st.divider()


# =========================
# TAB 3: Browse All Resumes
# =========================
with tab_browse:
    st.markdown("### All Indexed Resumes")
    st.markdown("View and manage all resumes currently stored in the Endee index.")

    if st.button("🔄 Refresh List", use_container_width=True):
        pass  # Triggers rerun

    resumes = list_all_resumes(endee_url, auth_token)

    if resumes is None:
        st.warning("Could not connect to Endee. Is the server running?")
    elif len(resumes) == 0:
        st.info("No resumes indexed yet. Go to the Upload tab to add some!")
    else:
        st.markdown(f"**Total resumes indexed: {len(resumes)}**")

        for resume in resumes:
            meta = resume.get("meta", {})
            filename = meta.get("filename", "Unknown")
            rid = resume.get("id", "N/A")

            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"📄 **{filename}** — `{rid}`")
            with col2:
                if st.button("🗑️ Delete", key=f"del_{rid}"):
                    ok, msg = delete_resume(endee_url, auth_token, rid)
                    if ok:
                        st.success(f"Deleted {filename}")
                        st.rerun()
                    else:
                        st.error(msg)


# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.divider()
st.markdown(
    """
    <div style="text-align:center; color:#999; font-size:0.85rem;">
        Built with ❤️ using <strong>Endee</strong> Vector Database,
        <strong>Sentence Transformers</strong>, and <strong>Streamlit</strong>
        <br/>
        <a href="https://github.com/endee-io/endee" target="_blank">Endee GitHub</a> •
        <a href="https://docs.endee.io" target="_blank">Endee Docs</a>
    </div>
    """,
    unsafe_allow_html=True,
)
