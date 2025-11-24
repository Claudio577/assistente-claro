from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def processar_documentos():
    embeddings = OpenAIEmbeddings()

    # Lista de PDFs
    arquivos = [
        "dados/politica_rh.pdf",
        "dados/manual_ti.pdf",
        "dados/onboarding.pdf"
    ]

    docs = []
    for arquivo in arquivos:
        loader = PyPDFLoader(arquivo)
        docs.extend(loader.load())

    # Quebra dos textos
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    textos = splitter.split_documents(docs)

    # Criação da base vetorial
    db = Chroma.from_documents(
        textos,
        embeddings,
        collection_name="claro_base"
    )

    print("📦 Documentos indexados com sucesso!")

if __name__ == "__main__":
    processar_documentos()

