import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import ollama
import os
import re
import pdfplumber
import pandas as pd

st.set_page_config(page_title="Askra", page_icon="✦", layout="centered")

# --- Custom styling: ChatGPT-like, subtle, professional ---
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
    }
    #MainMenu, footer, header {visibility: hidden;}

    .askra-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .askra-logo {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        font-size: 1.4rem;
        margin-bottom: 0.6rem;
    }
    .askra-header h1 {
        font-size: 1.7rem;
        font-weight: 600;
        margin: 0.4rem 0 0.2rem 0;
        color: #111827;
    }
    .askra-header p {
        color: #9ca3af;
        font-size: 0.9rem;
        margin: 0;
    }

    .stTextInput input {
        border-radius: 24px;
        border: 1px solid #e5e7eb;
        padding: 0.9rem 1.2rem;
        font-size: 0.95rem;
        background-color: #f7f7f8;
        color: #111827;
        box-shadow: none;
    }
    .stTextInput input:focus {
        border-color: #a5a6f6;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.08);
    }

    /* Chat-style message bubbles */
    .chat-bubble-user {
        background-color: #f3f4f6;
        color: #111827;
        border-radius: 16px 16px 4px 16px;
        padding: 0.9rem 1.2rem;
        margin: 1.2rem 0 0.6rem auto;
        max-width: 80%;
        font-size: 0.95rem;
        text-align: right;
    }
    .chat-bubble-assistant {
        background-color: #faf9fb;
        border: 1px solid #eeecf3;
        border-radius: 16px 16px 16px 4px;
        padding: 1.1rem 1.3rem;
        margin: 0 auto 0.6rem 0;
        max-width: 85%;
        font-size: 0.97rem;
        line-height: 1.65;
        color: #1f2937;
    }
    .assistant-label {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.75rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }

    .stExpander {
        border-radius: 10px;
        border: 1px solid #f0eef5 !important;
        background-color: #fdfcfe;
    }

    section[data-testid="stSidebar"] {
        background-color: #fbfafc;
        border-right: 1px solid #f0eef5;
    }
    section[data-testid="stSidebar"] h2 {
        color: #4f46e5;
        font-size: 1rem;
        font-weight: 600;
    }
    .file-badge {
        background-color: #f6f5fb;
        border: 1px solid #eeecf7;
        border-radius: 8px;
        padding: 0.45rem 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.83rem;
        color: #4b5563;
    }
    .stButton button {
        border-radius: 10px;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stButton button:hover {
        opacity: 0.92;
    }
</style>
""", unsafe_allow_html=True)

DOCS_FOLDER = "docs"
os.makedirs(DOCS_FOLDER, exist_ok=True)


# --- Extraction functions ---
def extract_pdf_text(filepath):
    text_parts = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    for row in table:
                        row_text = " | ".join(str(cell) for cell in row if cell)
                        if row_text.strip():
                            text_parts.append(row_text)
            else:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_csv_text(filepath):
    df = pd.read_csv(filepath)
    rows_as_text = []
    for _, row in df.iterrows():
        row_text = ", ".join(f"{col}: {row[col]}" for col in df.columns)
        rows_as_text.append(row_text)
    return "\n".join(rows_as_text)


def chunk_text(text, chunk_size=800, overlap=300):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def chunk_by_structure(text, max_chunk_size=1200):
    heading_pattern = r'\n(?=[A-Z][A-Za-z &/]{2,40}\n)'
    rough_sections = re.split(heading_pattern, text)

    chunks = []
    for section in rough_sections:
        section = section.strip()
        if not section:
            continue
        if len(section) > max_chunk_size:
            chunks.extend(chunk_text(section, chunk_size=800, overlap=300))
        else:
            chunks.append(section)

    if len(chunks) <= 1 and len(text) > max_chunk_size:
        return chunk_text(text)

    return chunks


def index_single_file(filepath, filename, collection, embedder):
    if filename.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    elif filename.endswith(".pdf"):
        text = extract_pdf_text(filepath)
    elif filename.endswith(".csv"):
        text = extract_csv_text(filepath)
    else:
        return 0

    chunks = chunk_by_structure(text)

    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk).tolist()
        collection.add(
            ids=[f"{filename}_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": filename}]
        )
    return len(chunks)


@st.cache_resource
def load_resources():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="askra_docs")
    return embedder, reranker, collection


embedder, reranker, collection = load_resources()


def rebuild_bm25_index(collection):
    all_data = collection.get()
    all_chunks = all_data["documents"]
    all_sources = [m["source"] for m in all_data["metadatas"]]
    tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_chunks) if all_chunks else None
    return bm25, all_chunks, all_sources


# --- Search + answer functions ---
def vector_search(question, top_k=25):
    query_embedding = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return list(zip(chunks, sources))


def keyword_search(question, bm25, all_chunks, all_sources, top_k=25):
    if bm25 is None:
        return []
    tokenized_query = question.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(all_chunks[i], all_sources[i]) for i in top_indices if scores[i] > 0]


def hybrid_search(question, bm25, all_chunks, all_sources):
    vector_results = vector_search(question)
    keyword_results = keyword_search(question, bm25, all_chunks, all_sources)
    seen = set()
    merged = []
    for chunk, source in vector_results + keyword_results:
        if chunk not in seen:
            seen.add(chunk)
            merged.append((chunk, source))
    return merged


def rerank(question, candidates, top_n=8):
    if not candidates:
        return []
    pairs = [[question, chunk] for chunk, source in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [(chunk, source, score) for (chunk, source), score in ranked[:top_n]]


def build_prompt(question, chunks):
    context = "\n\n".join(chunks)
    return f"""Answer the question using ONLY the context below.
The context may contain overlapping or repeated fragments from the same source — read all of it carefully and combine information across fragments rather than relying on just one.
Context may contain content from different sections of a document (e.g. Projects, Certifications, Experience, Education) — pay close attention to section headers and labels in the text, and only include items that are explicitly under the correct section for the question asked.
List each relevant item as a bullet point. Do not state a total count in your opening sentence — just present the list.
If the answer isn't in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""


def ask(question, bm25, all_chunks, all_sources):
    candidates = hybrid_search(question, bm25, all_chunks, all_sources)
    ranked = rerank(question, candidates, top_n=8)

    with st.expander("🔍 Debug: retrieved candidates"):
        for chunk, source, score in ranked:
            st.text(f"[{score:.3f}] [{source}] {chunk[:80]}")

    relevant = ranked

    if not relevant:
        return "I don't know based on the provided documents.", []

    filtered_chunks = [c for c, s, score in relevant]
    prompt = build_prompt(question, filtered_chunks)
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"], relevant


# --- UI ---
st.markdown("""
<div class="askra-header">
    <div class="askra-logo">✦</div>
    <h1>Askra</h1>
    <p>Ask questions about your own documents — answered locally and privately</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or CSV",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            filepath = os.path.join(DOCS_FOLDER, uploaded_file.name)
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())

        if st.button("Index uploaded files"):
            with st.spinner("Indexing..."):
                for uploaded_file in uploaded_files:
                    filepath = os.path.join(DOCS_FOLDER, uploaded_file.name)
                    num_chunks = index_single_file(filepath, uploaded_file.name, collection, embedder)
                    st.success(f"Indexed {uploaded_file.name}: {num_chunks} chunks")
            st.cache_resource.clear()
            st.rerun()

    st.divider()
    existing_files = [f for f in os.listdir(DOCS_FOLDER) if not f.startswith(".")]
    if existing_files:
        st.caption(f"{len(existing_files)} document(s)")
        for f in existing_files:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f'<div class="file-badge">{f}</div>', unsafe_allow_html=True)
            with col2:
                if st.button("🗑️", key=f"delete_{f}", help=f"Delete {f}"):
                    all_data = collection.get()
                    ids_to_delete = [
                        id_ for id_, meta in zip(all_data["ids"], all_data["metadatas"])
                        if meta["source"] == f
                    ]
                    if ids_to_delete:
                        collection.delete(ids=ids_to_delete)

                    filepath = os.path.join(DOCS_FOLDER, f)
                    if os.path.exists(filepath):
                        os.remove(filepath)

                    st.success(f"Deleted {f}")
                    st.rerun()
    else:
        st.caption("No documents yet")

bm25, all_chunks, all_sources = rebuild_bm25_index(collection)

question = st.text_input(
    "Ask a question",
    placeholder="Ask a question about your documents...",
    label_visibility="collapsed"
)

if question:
    st.markdown(f'<div class="chat-bubble-user">{question}</div>', unsafe_allow_html=True)

    with st.spinner("Thinking..."):
        answer, sources = ask(question, bm25, all_chunks, all_sources)

    st.markdown(f"""
    <div class="chat-bubble-assistant">
        <div class="assistant-label">✦ Askra</div>
        {answer}
    </div>
    """, unsafe_allow_html=True)

    if sources:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Sources")
        for i, (chunk, src, score) in enumerate(sources):
            with st.expander(f"{src} · relevance {score:.2f}"):
                st.write(chunk)
    else:
        st.caption("No sufficiently relevant chunks found in your documents.")