import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

st.title("Assistente Interno da Claro - Protótipo")
st.write("Pergunte sobre RH, TI ou documentos internos.")

# Modelo
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Embeddings locais
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 🔵 FUNÇÃO PARA RECRIAR A BASE VETORIAL DIRETO NO STREAMLIT
def atualizar_base():
    st.write("🔄 Atualizando base vetorial...")

    arquivos = [
        "dados/politica_rh.pdf",
        "dados/manual_ti.pdf",
        "dados/onboarding.pdf"
    ]

    docs = []
    for arquivo in arquivos:
        if os.path.exists(arquivo):
            loader = PyPDFLoader(arquivo)
            docs.extend(loader.load())
        else:
            st.error(f"Arquivo não encontrado: {arquivo}")
            return
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    textos = splitter.split_documents(docs)

    # Recria a base do zero
    Chroma.from_documents(
        textos,
        embeddings,
        collection_name="claro_base"
    )

    st.success("✅ Base vetorial atualizada com sucesso!")

# 🔵 BOTÃO PARA ATUALIZAR A BASE
if st.button("🔄 Atualizar base vetorial (rodar ingest)"):
    atualizar_base()

# Carrega a base existente (ou a nova após recriação)
db = Chroma(collection_name="claro_base", embedding_function=embeddings)

pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    consulta_expandida = (
        f"{pergunta} jornada horas semana carga horaria expediente beneficios RH TI Claro"
    )

    docs_list = db.similarity_search(consulta_expandida, k=5)

    contexto = "\n\n".join([d.page_content for d in docs_list])

    prompt = f"""
Você é um assistente interno da Claro.
Responda APENAS com base no CONTEXTO abaixo.

Se a informação não estiver no contexto, diga:
"Não encontrei essa informação nos documentos internos."

-----------------------
CONTEXTO:
{contexto}
-----------------------

PERGUNTA:
{pergunta}

Responda de forma clara e objetiva.
"""

    resposta = llm.invoke(prompt)

    st.write("### Resposta")
    st.write(resposta.content)

    with st.expander("Documentos utilizados"):
        for d in docs_list:
            st.write(d.page_content[:400] + "...")

