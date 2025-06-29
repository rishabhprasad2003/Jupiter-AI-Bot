import streamlit as st
import torch
import faiss
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Load Retriever & FAISS
@st.cache_resource
def load_index():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("faq_index.faiss")
    with open("faq_texts.pkl", "rb") as f:
        texts = pickle.load(f)
    return model, index, texts

retriever_model, index, faq_texts = load_index()

# Load LLM
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceH4/zephyr-7b-beta",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model()

# Retrieval
def retrieve_context(query, top_k=3):
    query_embedding = retriever_model.encode([query])
    scores, indices = index.search(query_embedding, top_k)
    return [faq_texts[i] for i in indices[0]]

# Generation
def generate_response(user_query):
    context_chunks = retrieve_context(user_query)
    context = "\n\n".join(context_chunks)

    prompt = f"""<|system|>
You are a helpful and friendly assistant for the Jupiter finance app. Use the context below to help the user. 
If the question can't be answered with the given context, respond:
"I'm not sure about that. Please contact Jupiter support."

<|user|>
Context:
{context}

Question:
{user_query}

<|assistant|>"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("<|assistant|>")[-1].strip()

# Streamlit UI
st.title("📘 Jupiter FAQ Chatbot")
st.markdown("Ask questions about Jupiter's services. The bot retrieves relevant FAQs and generates helpful replies.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    st.chat_message("user").markdown(msg["user"])
    st.chat_message("assistant").markdown(msg["bot"])

user_input = st.chat_input("Type your question...")

if user_input:
    with st.spinner("Thinking..."):
        bot_reply = generate_response(user_input)

    st.chat_message("user").markdown(user_input)
    st.chat_message("assistant").markdown(bot_reply)

    st.session_state.chat_history.append({"user": user_input, "bot": bot_reply})
