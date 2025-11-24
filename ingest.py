# ingest.py
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

def processar_documentos():
    embeddings = OpenAIEmbeddings()

    # Carrega documentos
    docs = []
    for arquivo in ["dados/politicas_rh.pdf", "dados/manual_ti.pdf"]:
        loader = PyPDFLoader(arquivo)
        docs.extend(loader.load())

    # Divide em pedaços menores
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    textos = splitter.split_documents(docs)

    # Cria vetor store
    db = Chroma.from_documents(textos, embeddings, collection_name="claro_base")
    print("Documentos indexados com sucesso!")

if __name__ == "__main__":
    processar_documentos()
