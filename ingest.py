import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Pasta segura para Streamlit Cloud
FAISS_DIR = "/mount/data/base_faiss"

def processar_documentos():
    # Criar pasta segura
    os.makedirs(FAISS_DIR, exist_ok=True)

    arquivos = [
        "dados/politica_rh.pdf",
        "dados/manual_ti.pdf",
        "dados/onboarding.pdf"
    ]

    docs = []
    for a in arquivos:
        loader = PyPDFLoader(a)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Criar FAISS
    db = FAISS.from_documents(chunks, embeddings)

    # Salvar no local permitido
    db.save_local(FAISS_DIR)

    print("FAISS salva em:", FAISS_DIR)
