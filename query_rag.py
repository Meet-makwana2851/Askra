import chromadb
from sentence_transformers import SentenceTransformer
import ollama

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_docs")

def retrieve(question, top_k=8):
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

    print("\n--- Raw distances (debug) ---")
    for c, d in zip(chunks, distances):
        print(f"distance={d:.4f}  {c[:60]}...")

    relevant = [(c, d) for c, d in zip(chunks, distances) if d < distance_threshold]

    if not relevant:
        return "I don't know based on the provided documents.", [], []

    filtered_chunks = [c for c, d in relevant]
    filtered_distances = [d for c, d in relevant]
    prompt = build_prompt(question, filtered_chunks)

    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"], filtered_chunks, filtered_distances

if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or 'quit'): ")
        if question.lower() == "quit":
            break
        answer, used_chunks, distances = ask(question)
        print("\n--- Answer ---")
        print(answer)
        print("\n--- Sources used ---")
        for i, (c, d) in enumerate(zip(used_chunks, distances)):
            print(f"[{i+1}] distance={d:.4f}  {c[:80]}...")