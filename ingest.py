# ingest.py - FAISS VERSION

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import os

def processar_documentos():
    os.makedirs("faiss_store", exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    arquivos = [
        "dados/politica_rh.pdf",
        "dados/manual_ti.pdf",
        "dados/onboarding.pdf"
    ]

    docs = []
    for arquivo in arquivos:
        loader = PyPDFLoader(arquivo)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    textos = splitter.split_documents(docs)

    # Criar o índice FAISS
    store = FAISS.from_documents(textos, embeddings)

    # Salvar em disco
    store.save_local("faiss_store")

    print("✔️ Vetores FAISS gerados e salvos em /faiss_store")


if __name__ == "__main__":
    processar_documentos()
