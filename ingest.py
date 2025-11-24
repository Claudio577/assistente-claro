from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def processar_documentos():

    # Criar pasta da base FAISS se não existir
    os.makedirs("base_faiss", exist_ok=True)

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

    # Criar FAISS
    db = FAISS.from_documents(textos, embeddings)

    # Salvar
    db.save_local("base_faiss")

    print("✅ Base FAISS criada com sucesso!")

if __name__ == "__main__":
    processar_documentos()
