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

uploads = load_index()  # mapping file_id -> filename

# Initialize chat history
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
    result = collection.get(where={"file_id": fid}, include=["documents", "metadatas"])
    return result.get("documents", []), result.get("metadatas", [])


def llm_query(model, prompt):
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False, "max_tokens": MAX_TOKENS}
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        st.error(f"LLM error: {e}")
        st.stop()

# ---- STREAMLIT UI ----
st.set_page_config(page_title="AI File Manager", layout="wide")
header = st.empty()
header.title("🧠 AI File Manager")

# Sidebar: Uploaded Files & Chat History
with st.sidebar:
    st.header("📂 Uploaded Files")
    filenames = list(uploads.values())
    selected_filenames = st.multiselect("Select file(s)", options=filenames)
    selected_file_ids = [fid for fid, name in uploads.items() if name in selected_filenames]

    if selected_filenames and st.button("❌ Delete Selected Files"):
        for fname in selected_filenames:
            collection.delete(where={"filename": fname})
        for fid in selected_file_ids:
            uploads.pop(fid, None)
        save_index(uploads)
        st.success("Deleted selected files and their chunks. Please refresh to update list.")

    st.markdown("---")
    st.header("🗨️ Chat History")
    if st.button("🗑️ Clear Chat History"):
        st.session_state['chat_history'] = []
        st.success("Chat history cleared.")
    history = st.session_state['chat_history']
    if history:
        hist_indices = st.multiselect(
            "View past questions", options=list(range(len(history))),
            format_func=lambda i: history[i]['query']
        )
        for i in hist_indices:
            st.markdown(f"**Q:** {history[i]['query']}")
            st.markdown(f"**A:** {history[i]['answer']}")
    else:
        st.info("No chat history yet.")

# Main Panel: Model & Stats
g1, g2, g3 = st.columns([2,1,1])
with g1:
    model = st.selectbox("LLM Model", ["llama3.2:latest", "mistral:latest", "phi4:latest"], index=2)
with g2:
    st.metric("Files", len(uploads))
with g3:
    st.metric("Chunks", collection.count())

# OCR Settings
with st.expander("⚙️ OCR Settings"):
    ocr_langs = st.multiselect("OCR Languages", ["eng","hin","ben","tam"], default=["eng"])

# File Upload & Processing
uploaded = st.file_uploader("Upload Documents", accept_multiple_files=True, key="file_uploader")
process_button = st.button("Process & Store Files")
if process_button and uploaded:
    header.title("⏳ Processing...")
    with st.spinner("Processing and storing files... Please wait..."):
        for file in uploaded:
            if file.name in uploads.values():
                st.warning(f"{file.name} already uploaded; skipping.")
                continue
            fid = rand_id()
            chunks = ocr_parse(file, ocr_langs)
            insert_chunks(chunks, fid, file.name)
            uploads[fid] = file.name
        save_index(uploads)
    # reset uploader to remove processed files
    st.session_state['file_uploader'] = None
    header.title("🧠 AI File Manager")
    st.success("Files processed and stored.")

# Summarize Selected Files
if selected_file_ids and st.button("📝 Summarize Selected Files"):
    header.title("⏳ Summarizing...")
    with st.spinner("Summarizing selected files... Please wait..."):
        for fid in selected_file_ids:
            fname = uploads.get(fid)
            docs, _ = fetch_chunks(fid)
            text = "".join(docs)
            prompt = (
                f"Provide a detailed summary (max {MAX_TOKENS} tokens) for the document '{fname}':" + text
            )
            summ = llm_query(model, prompt)
            st.markdown(f"### Summary for {fname}")
            st.write(summ)
            st.download_button(
                label=f"Download Summary for {fname}",
                data=summ,
                file_name=f"summary_{fname}.txt",
                key=f"dl_{fid}"
            )
    header.title("🧠 AI File Manager")

# Chatbot: RAG & General
st.subheader("💬 Chatbot")
mode = st.radio("Mode", ["RAG", "General"], horizontal=True)
query = st.text_input("Your question...")
if query:
    header.title("⏳ Generating Answer...")
    with st.spinner("Running query... Please wait..."):
        if mode == "RAG":
            retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
            if selected_file_ids:
                retriever.search_kwargs['filter'] = {"file_id": {"$in": selected_file_ids}}
            qa = RetrievalQA.from_llm(llm=ChatOllama(model=model), retriever=retriever, return_source_documents=True)
            result = qa({"query": query})
            answer = result["result"]
            sources = result.get("source_documents", [])[:TOP_K]
            st.subheader("🧠 Answer")
            st.write(answer)
            st.subheader("🔍 Top Chunks Used")
            for doc in sources:
                idx = doc.metadata.get("chunk_index")
                fname = doc.metadata.get("filename")
                with st.expander(f"Chunk {idx} (File: {fname})"):
                    st.write(doc.page_content)
                    st.json(doc.metadata)
        else:
            answer = llm_query(model, query)
            st.subheader("🧠 Answer")
            st.write(answer)
        st.session_state['chat_history'].append({"query": query, "answer": answer})
    header.title("🧠 AI File Manager")
