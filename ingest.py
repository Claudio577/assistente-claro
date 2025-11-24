import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_PATH = "dados"
CHROMA_PATH = "chroma"

def processar_documentos():
    # Embeddings locais
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Carregar PDFs
    arquivos = [
        "dados/politica_rh.pdf",
        "dados/manual_ti.pdf",
        "dados/onboarding.pdf"
    ]

    docs = []
    for arquivo in arquivos:
        loader = PyPDFLoader(arquivo)
        docs.extend(loader.load())

    # Dividir em chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    textos = splitter.split_documents(docs)

    # Criar vetorstore com persistência
    Chroma.from_documents(
        textos,
        embeddings,
        collection_name="claro_base",
        persist_directory=CHROMA_PATH
    )

    print("📦 Base vetorial atualizada com sucesso!")

if __name__ == "__main__":
    processar_documentos()
