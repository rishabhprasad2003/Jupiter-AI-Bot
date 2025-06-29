# app.py

import streamlit as st
import faiss
import pickle
from utils import load_model, format_prompt, generate_answer

# Load embeddings
with open("faq_texts.pkl", "rb") as f:
    faq_texts = pickle.load(f)

index = faiss.read_index("faq_index.faiss")

# Load model only once
@st.cache_resource
def get_model():
    return load_model()

tokenizer, model = get_model()

# UI
st.title("🧠 Jupiter AI FAQ Bot")
user_query = st.text_input("Ask a question:")

if user_query:
    # Embed the question and search (you already have this logic)
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = embedder.encode([user_query])
    D, I = index.search(query_vec, k=3)
    context = "\n".join([faq_texts[i] for i in I[0]])

    # Format prompt & get answer
    prompt = format_prompt(context, user_query)
    answer = generate_answer(model, tokenizer, prompt)
    st.markdown("### 🤖 Answer:")
    st.write(answer)
