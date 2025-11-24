import shutil
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def processar_documentos():

    # 1 — limpar base antiga
    shutil.rmtree("chroma", ignore_errors=True)

    # 2 — carregar embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3 — PDFs na pasta dados/
    arquivos = [
        "dados/politica_rh.pdf",
        "dados/manual_ti.pdf",
        "dados/onboarding.pdf"
    ]

    docs = []
    for arquivo in arquivos:
        loader = PDFPlumberLoader(arquivo)
        docs.extend(loader.load())

    # 4 — dividir em chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    textos = splitter.split_documents(docs)

    # 5 — criar base vetorial persistida
    Chroma.from_documents(
        documentos=textos,
        embedding=embeddings,
        persist_directory="chroma",
        collection_name="claro_base"
    )

    print("📦 Base vetorial gerada com sucesso!")


if __name__ == "__main__":
    processar_documentos()
