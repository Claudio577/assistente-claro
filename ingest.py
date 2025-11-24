from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def processar_documentos():
    pasta_dados = "dados"
    arquivos = [f"{pasta_dados}/{f}" for f in os.listdir(pasta_dados) if f.endswith(".pdf")]

    if not arquivos:
        raise Exception("Nenhum PDF encontrado na pasta 'dados'.")

    docs = []
    for arquivo in arquivos:
        loader = PyPDFLoader(arquivo)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    textos = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_documents(textos, embeddings)
    db.save_local("base_faiss")

    print("Base vetorial FAISS criada com sucesso!")
