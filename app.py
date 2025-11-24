import streamlit as st
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from ingest import processar_documentos
import tempfile

FAISS_DIR = os.path.join(tempfile.gettempdir(), "base_faiss")

db = FAISS.load_local(
    FAISS_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

FAISS_DIR = "/mount/data/base_faiss"

load_dotenv()

st.title("Assistente Interno da Claro - Protótipo")
st.write("Pergunte sobre RH, TI ou documentos internos.")

# -------- BOTÃO DE RECRIAÇÃO DA BASE --------
if st.button("🔄 Atualizar base vetorial (rodar ingest)"):
    with st.spinner("Processando documentos e criando vetores..."):
        processar_documentos()
    st.success("Base vetorial criada com sucesso!")

# -------- VERIFICAR A BASE --------
if not os.path.exists(FAISS_DIR):
    st.warning("⚠ A base vetorial ainda não foi criada. Clique acima para rodar o ingest.")
    st.stop()

# -------- CARREGAR BASE --------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)

# -------- MODELO --------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -------- CAIXA DE PERGUNTA --------
pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    resultados = db.similarity_search(pergunta, k=5)

    if len(resultados) == 0:
        st.error("❌ Não encontrei essa informação nos documentos internos.")
        st.stop()

    contexto = "\n\n".join([doc.page_content for doc in resultados])

    prompt = f"""
    Responda APENAS com base nos documentos abaixo:

    {contexto}

    Pergunta: {pergunta}

    Se a resposta não estiver nos documentos, diga: 'não consta'.
    """

    resposta = llm.invoke(prompt)

    st.write("### Resposta:")
    st.write(resposta.content)

    with st.expander("Documentos utilizados"):
        for doc in resultados:
            st.write(doc.page_content[:300] + "...")
