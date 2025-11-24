from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def processar_documentos():
    # Embeddings 100% locais (sem OpenAI)
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

    # Dividir textos
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    textos = splitter.split_documents(docs)

    # Gerar base vetorial em Chroma
    db = Chroma.from_documents(
        textos,
        embeddings,
        collection_name="claro_base"
    )

    print("📦 Documentos indexados com sucesso usando embeddings locais!")

if __name__ == "__main__":
    processar_documentos()
