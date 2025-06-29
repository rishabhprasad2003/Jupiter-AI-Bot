# utils.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

def format_prompt(context, user_query):
    return f"""<|system|>
You are a helpful assistant for the Jupiter Money app. Use the context below to answer.
If you’re not sure, say: "I'm not sure. Please contact Jupiter support."

<|user|>
Context:
{context}

Question:
{user_query}

<|assistant|>"""

def generate_answer(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("<|assistant|>")[-1].strip()
