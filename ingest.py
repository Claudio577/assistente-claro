import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def processar_documentos():

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
        chunk_overlap=200
    )
    textos = splitter.split_documents(docs)

    # ---- FAISS ----
    db = FAISS.from_documents(textos, embeddings)
    db.save_local("base_faiss")

    print("Base vetorial criada com sucesso!")

