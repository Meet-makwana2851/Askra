import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import ollama
import os
from pypdf import PdfReader
import pandas as pd

st.set_page_config(page_title="Askra", page_icon="📚")

# --- Custom styling ---
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
    }
    .askra-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .askra-header h1 {
        font-size: 2.8rem;
        margin-bottom: 0;
        background: linear-gradient(90deg, #7c3aed, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .askra-header p {
        color: #6b7280;
        font-size: 1rem;
    }
    .stTextInput input {
        border-radius: 12px;
        border: 1px solid #d1d5db;
        padding: 0.8rem 1rem;
        font-size: 1rem;
        background-color: #ffffff;
        color: #1f2937;
    }
    .answer-card {
        background-color: #f9f7ff;
        border: 1px solid #e5deff;
        border-radius: 14px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 2px 6px rgba(124, 58, 237, 0.08);
    }
    .answer-card h4 {
        color: #7c3aed;
        margin-top: 0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .answer-card p {
        color: #1f2937;
    }
    .stExpander {
        border-radius: 10px;
        border: 1px solid #e5e7eb !important;
        background-color: #ffffff;
    }
    section[data-testid="stSidebar"] {
        background-color: #faf9fc;
        border-right: 1px solid #e5e7eb;
    }
    section[data-testid="stSidebar"] h2 {
        color: #7c3aed;
        font-size: 1.1rem;
    }
    .file-badge {
        background-color: #f5f3ff;
        border: 1px solid #e5deff;
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
        color: #4b5563;
    }
    .stButton button {
        border-radius: 10px;
        background: linear-gradient(90deg, #7c3aed, #a855f7);
        color: white;
        border: none;
        font-weight: 600;
    }
    .stButton button:hover {
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

DOCS_FOLDER = "docs"
os.makedirs(DOCS_FOLDER, exist_ok=True)


# --- Extraction functions ---
def extract_pdf_text(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def extract_csv_text(filepath):
    df = pd.read_csv(filepath)
    rows_as_text = []
    for _, row in df.iterrows():
        row_text = ", ".join(f"{col}: {row[col]}" for col in df.columns)
        rows_as_text.append(row_text)
    return "\n".join(rows_as_text)


def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
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

    chunks = chunk_text(text)
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
def vector_search(question, top_k=15):
    query_embedding = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return list(zip(chunks, sources))


def keyword_search(question, bm25, all_chunks, all_sources, top_k=15):
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


def rerank(question, candidates, top_n=5):
    if not candidates:
        return []
    pairs = [[question, chunk] for chunk, source in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [(chunk, source, score) for (chunk, source), score in ranked[:top_n]]


def build_prompt(question, chunks):
    context = "\n\n".join(chunks)
    return f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""


def ask(question, bm25, all_chunks, all_sources, relevance_threshold=0.0):
    candidates = hybrid_search(question, bm25, all_chunks, all_sources)
    ranked = rerank(question, candidates)
    relevant = [(c, s, score) for c, s, score in ranked if score > relevance_threshold]

    if not relevant:
        return "I don't know based on the provided documents.", []

    filtered_chunks = [c for c, s, score in relevant]
    prompt = build_prompt(question, filtered_chunks)
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"], relevant


# --- UI ---
st.markdown("""
<div class="askra-header">
    <h1>📚 Askra</h1>
    <p>Your local, private knowledge assistant — runs entirely on your Mac</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or CSV files",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True
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
        st.caption(f"📂 {len(existing_files)} document(s) indexed")
        for f in existing_files:
            st.markdown(f'<div class="file-badge">📄 {f}</div>', unsafe_allow_html=True)
    else:
        st.caption("No documents yet — upload one above")

bm25, all_chunks, all_sources = rebuild_bm25_index(collection)

question = st.text_input("Ask a question about your documents:")

if question:
    with st.spinner("Thinking..."):
        answer, sources = ask(question, bm25, all_chunks, all_sources)

    st.markdown(f"""
    <div class="answer-card">
        <h4>Answer</h4>
        <p style="font-size: 1.05rem; line-height: 1.6;">{answer}</p>
    </div>
    """, unsafe_allow_html=True)

    if sources:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📎 Sources")
        for i, (chunk, src, score) in enumerate(sources):
            with st.expander(f"📄 {src} · relevance {score:.2f}"):
                st.write(chunk)
    else:
        st.caption("No sufficiently relevant chunks found in your documents.")