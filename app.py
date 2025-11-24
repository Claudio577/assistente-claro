import streamlit as st
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# IMPORTAÇÃO CORRETA DO SPLITTER (nova versão)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest import processar_documentos


# ---------------------
# INICIALIZAÇÃO
# ---------------------
load_dotenv()

st.title("Assistente Interno da Claro - Protótipo")
st.write("Pergunte sobre RH, TI ou documentos internos.")


# ---------------------
# BOTÃO PARA RODAR INGEST
# ---------------------
if st.button("🔄 Atualizar base vetorial (rodar ingest)"):
    with st.spinner("Processando documentos e criando base vetorial..."):
        processar_documentos()
    st.success("Base vetorial criada com sucesso!")


# ---------------------
# CHECAR SE A BASE EXISTE
# ---------------------
if not os.path.exists("base_faiss"):
    st.warning("⏳ Base vetorial não encontrada. Clique no botão acima para rodar o ingest.")
    st.stop()


# ---------------------
# CARREGAR BASE FAISS
# ---------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local(
    "base_faiss",
    embeddings,
    allow_dangerous_deserialization=True
)


# ---------------------
# CONFIG OPENAI
# ---------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# ---------------------
# PERGUNTA DO USUÁRIO
# ---------------------
pergunta = st.text_input("Digite sua pergunta:")


if pergunta:

    # buscar nos vetores
    resultados = db.similarity_search(pergunta, k=5)

    if len(resultados) == 0:
        st.error("❌ Não encontrei essa informação nos documentos internos.")
        st.stop()

    # construir contexto
    contexto = "\n\n".join([doc.page_content for doc in resultados])

    prompt = f"""
    Você é um assistente interno da Claro.

    Responda APENAS com base nos documentos abaixo:

    {contexto}

    Pergunta: {pergunta}

    Se a resposta não estiver clara nos documentos, diga:
    "Não encontrei essa informação nos documentos internos."
    """

    resposta = llm.invoke(prompt)

    st.write("### Resposta")
    st.write(resposta.content)

    with st.expander("Documentos utilizados"):
        for doc in resultados:
            st.write(doc.page_content[:400] + "...")
