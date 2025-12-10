# ingest_database.py
# Reads PDFs → splits into chunks → converts to embeddings → saves FAISS index.

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load env vars (not strictly needed for embeddings, but safe)
load_dotenv()

# configuration
DATA_PATH = "data"          # folder with your PDFs
FAISS_PATH = "faiss_index"  # folder where FAISS will be stored

# 1. Hugging Face embeddings (fully local)
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load PDFs
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()

# 3. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(raw_documents)

# 4. Build FAISS vector store and save it
vector_store = FAISS.from_documents(chunks, embeddings_model)
vector_store.save_local(FAISS_PATH)

print("✅ Ingestion complete. FAISS index saved to:", FAISS_PATH)

