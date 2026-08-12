import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

st.set_page_config(page_title="My RAG Assistant")

@st.cache_resource
def load_resources():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="my_docs")
    return embedder, collection

embedder, collection = load_resources()

def retrieve(question, top_k=6):
    query_embedding = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0], results["distances"][0]

def build_prompt(question, chunks):
    context = "\n\n".join(chunks)
    return f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""

def ask(question, distance_threshold=1.6):
    chunks, distances = retrieve(question)
    relevant = [(c, d) for c, d in zip(chunks, distances) if d < distance_threshold]

    if not relevant:
        return "I don't know based on the provided documents.", []

    filtered_chunks = [c for c, d in relevant]
    prompt = build_prompt(question, filtered_chunks)

    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"], relevant

st.title("📚 My Local RAG Assistant")
question = st.text_input("Ask a question about your documents:")

if question:
    with st.spinner("Thinking..."):
        answer, sources = ask(question)

    st.subheader("Answer")
    st.write(answer)

    if sources:
        st.subheader("Sources used")
        for i, (chunk, dist) in enumerate(sources):
            with st.expander(f"Source {i+1} (distance: {dist:.4f})"):
                st.write(chunk)