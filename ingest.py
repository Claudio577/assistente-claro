from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import os

def processar_documentos():
    os.makedirs("vectorstore", exist_ok=True)

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

    db = FAISS.from_documents(textos, embeddings)
    db.save_local("vectorstore")   # <-- salva a base

    print("📦 Base vetorial FAISS atualizada com sucesso!")

if __name__ == "__main__":
    processar_documentos()
