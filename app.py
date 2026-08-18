import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama

st.set_page_config(page_title="Askra", page_icon="📚")

@st.cache_resource
def load_resources():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="askra_docs")
    return embedder, reranker, collection

embedder, reranker, collection = load_resources()

def retrieve(question, top_k=20):
    query_embedding = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return chunks, sources

def rerank(question, chunks, sources, top_n=5):
    pairs = [[question, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(chunks, sources, scores), key=lambda x: x[2], reverse=True)
    return ranked[:top_n]

def build_prompt(question, chunks):
    context = "\n\n".join(chunks)
    return f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""

def ask(question, relevance_threshold=0.0):
    chunks, sources = retrieve(question)
    ranked = rerank(question, chunks, sources)

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