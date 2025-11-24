# ingest.py

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

FAISS_DIR = "base_faiss"

def processar_documentos():
    os.makedirs(FAISS_DIR, exist_ok=True)

    documentos = []

    # Carrega todos os PDFs da pasta dados/
    for arquivo in os.listdir("dados"):
        if arquivo.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("dados", arquivo))
            documentos.extend(loader.load())

    # Divide os textos em chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documentos)

    # Cria embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Cria a base FAISS
    db = FAISS.from_documents(chunks, embeddings)

    # Salva localmente
    db.save_local(FAISS_DIR)
