# 🧠 Jupiter AI FAQ Chatbot

A locally deployed, LLM-powered chatbot that answers user queries by leveraging real customer questions and responses scraped from the Jupiter Money Community Help forum.

---

## 🔍 Project Overview

This project builds an intelligent chatbot using large language models (LLMs) to provide context-aware and human-friendly responses for customer support. It includes data scraping, preprocessing, semantic search, and answer generation with models like Zephyr-7B or Flan-T5.

---

## 🚀 Features

- 🔄 Scrapes 300+ FAQ threads with full questions and replies from [Jupiter Community](https://community.jupiter.money/c/help/27)
- 🧹 Structures raw data into clean Q&A pairs categorized for training and inference
- 🔍 Embeds and indexes questions using `SentenceTransformers` and `FAISS` for semantic similarity
- 🤖 Uses Zephyr-7B (local) or Flan-T5 (cloud) to generate accurate, natural responses
- 🖥️ Streamlit-based user interface for interactive querying
- 🧠 Gracefully handles unknown queries and rephrases answers

---

## 🧱 Tech Stack

- **Python**
- **Hugging Face Transformers** (Zephyr-7B, Flan-T5)
- **BeautifulSoup4** (for web scraping)
- **FAISS** (semantic search)
- **Sentence-Transformers**
- **Streamlit** (chat UI)
- **Torch** (model execution)

---

## 🗂️ Project Structure

