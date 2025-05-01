# unstructured_chatbot.py
# AI File Manager App - RAG & General LLM with Chat & Session History

import streamlit as st
import requests
import os
import json
import random
import string
from datetime import datetime

# ChromaDB imports
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chat_models import ChatOllama
from langchain.chains import RetrievalQA

# ---- CONFIG ----
UNSTRUCTURED_URL = "http://localhost:9500/general/v0/general"
OLLAMA_URL       = "http://localhost:11434/api/generate"
UPLOAD_INDEX     = "uploads_index.json"
VECTOR_DIR       = "chroma_data"
MAX_TOKENS       = 4096  # for summarization and chat
TOP_K            = 10    # top chunks for RAG

# ---- INITIALIZATION ----
# Ensure storage directory exists
os.makedirs(VECTOR_DIR, exist_ok=True)

# Persistent ChromaDB client
settings = Settings(anonymized_telemetry=False)
chroma_client = PersistentClient(
    path=VECTOR_DIR,
    settings=settings,
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE
)
collection = chroma_client.get_or_create_collection(name="docs")

# LangChain vectorstore wrapper
embedding_fn = OllamaEmbeddings(model="nomic-embed-text")
vectorstore  = Chroma(
    client_settings=settings,
    collection_name="docs",
    embedding_function=embedding_fn
)

# ---- STATE MANAGEMENT ----
def load_index():
    if os.path.exists(UPLOAD_INDEX):
        with open(UPLOAD_INDEX, "r") as f:
            return json.load(f)
    return {}

def save_index(idx):
    with open(UPLOAD_INDEX, "w") as f:
        json.dump(idx, f, indent=2)

uploads = load_index()

# Initialize session chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# ---- HELPERS ----
def rand_id(n=6):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))

def ocr_parse(file, langs):
    try:
        resp = requests.post(
            UNSTRUCTURED_URL,
            files={"files": (file.name, file.getvalue())},
            data={"ocr_languages": ','.join(langs)}
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"OCR error: {e}")
        st.stop()


def insert_chunks(chunks, fid, fname):
    texts, metas, ids = [], [], []
    ts = datetime.utcnow().isoformat()
    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "").strip()
        if text:
            texts.append(text)
            metas.append({
                "file_id": fid,
                "filename": fname,
                "chunk_index": idx,
                "timestamp": ts
            })
            ids.append(f"{fid}_{idx}")
    if texts:
        vectorstore.add_texts(texts, metadatas=metas, ids=ids)


def fetch_chunks(fid):
    res = collection.get(where={"file_id": fid})
    return res.get("documents", [])


def llm_query(model, prompt):
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False, "max_tokens": MAX_TOKENS}
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        st.error(f"LLM error: {e}")
        st.stop()

# ---- STREAMLIT UI ----
st.set_page_config(page_title="AI File Manager", layout="wide")
st.title("🧠 AI File Manager")

# Sidebar: Files and Chat History
with st.sidebar:
    st.header("📂 Uploaded Files")
    selected_files = st.multiselect(
        "Select file(s)",
        options=list(uploads.keys()),
        format_func=lambda fid: uploads[fid]
    )
    if selected_files and st.button("❌ Delete Selected Files"):
        for fid in selected_files:
            collection.delete(where={"file_id": fid})
            uploads.pop(fid, None)
        save_index(uploads)
        st.experimental_rerun()

    st.markdown("---")
    st.header("🗨️ Chat History")
    history = st.session_state['chat_history']
    if history:
        selected_hist = st.multiselect(
            "Select history entries", 
            options=list(range(len(history))),
            format_func=lambda i: history[i]['query']
        )
        for i in selected_hist:
            st.markdown(f"**Q:** {history[i]['query']}")
            st.markdown(f"**A:** {history[i]['answer']}")
    else:
        st.info("No chat history yet.")

# Main panel: Model & Stats
g1, g2, g3 = st.columns([2,1,1])
with g1:
    model = st.selectbox("LLM Model", ["llama3.2:latest", "mistral:latest", "phi4:latest"])
with g2:
    st.metric("Files", len(uploads))
with g3:
    st.metric("Chunks", collection.count())

# OCR settings
with st.expander("⚙️ OCR Settings"):
    ocr_langs = st.multiselect(
        "OCR Languages", ["eng","hin","ben","tam"],
        default=["eng"]
    )

# File upload & processing
uploaded = st.file_uploader("Upload Documents", accept_multiple_files=True)
if uploaded and st.button("Process & Store Files"):
    for file in uploaded:
        if file.name in uploads.values():
            st.warning(f"{file.name} already uploaded; skipping.")
            continue
        fid = rand_id()
        chunks = ocr_parse(file, ocr_langs)
        insert_chunks(chunks, fid, file.name)
        uploads[fid] = file.name
    save_index(uploads)
    st.success("Files processed and stored.")

# Summarize selected files
if selected_files:
    combined = []
    for fid in selected_files:
        fname = uploads[fid]
        st.subheader(f"Chunks for {fname}")
        ch = fetch_chunks(fid)
        st.text_area(f"Chunks - {fname}", "\n\n".join(ch), height=200)
        combined.extend(ch)
    if st.button("📝 Summarize Selected Files"):
        prompt_text = (
            f"Provide a detailed summary (max {MAX_TOKENS} tokens) for the following content:\n\n" +
            "\n\n".join(combined)
        )
        summary = llm_query(model, prompt_text)
        st.subheader("📝 Summary")
        st.write(summary)
        st.download_button("Download Summary", summary, file_name="summary.txt")

# Chatbot: RAG & General
st.subheader("💬 Chatbot")
mode = st.radio("Mode", ["RAG", "General"], horizontal=True)
query = st.text_input("Your question...")
if query:
    if mode == "RAG":
        retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        if selected_files:
            retriever.search_kwargs['filter'] = {"file_id": {"$in": selected_files}}
        qa = RetrievalQA.from_llm(
            llm=ChatOllama(model=model),
            retriever=retriever,
            return_source_documents=True
        )
        result = qa({"query": query})
        answer = result["result"]
        sources = result.get("source_documents", [])[:TOP_K]
        st.subheader("🧠 Answer")
        st.write(answer)
        st.subheader("🔍 Top Chunks Used")
        for doc in sources:
            st.markdown(f"- {doc.page_content}")
    else:
        answer = llm_query(model, query)
        st.subheader("🧠 Answer")
        st.write(answer)
    # Append to chat history
    st.session_state['chat_history'].append({"query": query, "answer": answer})
