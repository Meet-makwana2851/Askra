import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import ollama

st.set_page_config(page_title="Askra", page_icon="📚")

@st.cache_resource
def load_resources():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="askra_docs")

    # Load ALL chunks once to build the BM25 keyword index
    all_data = collection.get()
    all_chunks = all_data["documents"]
    all_sources = [m["source"] for m in all_data["metadatas"]]
    tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    return embedder, reranker, collection, bm25, all_chunks, all_sources

embedder, reranker, collection, bm25, all_chunks, all_sources = load_resources()

def vector_search(question, top_k=15):
    query_embedding = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return list(zip(chunks, sources))

def keyword_search(question, top_k=15):
    tokenized_query = question.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(all_chunks[i], all_sources[i]) for i in top_indices if scores[i] > 0]

def hybrid_search(question):
    vector_results = vector_search(question)
    keyword_results = keyword_search(question)

    # Merge and deduplicate (by chunk text)
    seen = set()
    merged = []
    for chunk, source in vector_results + keyword_results:
        if chunk not in seen:
            seen.add(chunk)
            merged.append((chunk, source))
    return merged

def rerank(question, candidates, top_n=5):
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

def ask(question, relevance_threshold=0.0):
    candidates = hybrid_search(question)
    ranked = rerank(question, candidates)

    relevant = [(c, s, score) for c, s, score in ranked if score > relevance_threshold]

    if not relevant:
        return "I don't know based on the provided documents.", []

    filtered_chunks = [c for c, s, score in relevant]
    prompt = build_prompt(question, filtered_chunks)

    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"], relevant

# --- UI ---
st.title("📚 Askra")
st.caption("Your local, private knowledge assistant — runs entirely on your Mac.")

question = st.text_input("Ask a question about your documents:")

if question:
    with st.spinner("Thinking..."):
        answer, sources = ask(question)

    st.subheader("Answer")
    st.write(answer)

    if sources:
        st.subheader("Sources used")
        for i, (chunk, src, score) in enumerate(sources):
            with st.expander(f"📄 {src} — Source {i+1} (relevance: {score:.4f})"):
                st.write(chunk)
    else:
        st.caption("No sufficiently relevant chunks found in your documents.")