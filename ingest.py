from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def processar_documentos():

    # Criar pasta da base caso não exista
    if not os.path.exists("base_faiss"):
        os.makedirs("base_faiss")

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

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    FAISS.from_documents(
        textos,
        embeddings
    ).save_local("base_faiss")

    print("Base FAISS criada com sucesso!")
