import streamlit as st
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from ingest import processar_documentos   # IMPORTAÇÃO CORRETA

load_dotenv()

st.title("Assistente Interno da Claro - Protótipo")
st.write("Pergunte sobre RH, TI ou documentos internos.")

# --- BOTÃO DE INGEST ---
if st.button("🔄 Atualizar base vetorial (rodar ingest)"):
    with st.spinner("Processando documentos e criando base vetorial..."):
        processar_documentos()
        st.success("Base vetorial criada com sucesso!")

# --------- CARREGAR BASE ----------
def carregar_base():
    if not os.path.exists("base_faiss"):
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("base_faiss", embeddings, allow_dangerous_deserialization=True)

db = carregar_base()

# --------- MODELO -----------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -------- PERGUNTA ----------
pergunta = st.text_input("Digite sua pergunta:")

if pergunta:

    if db is None:
        st.error("⚠️ Base vetorial ainda não foi criada. Clique no botão acima para rodar o ingest.")
        st.stop()

    docs = db.similarity_search(pergunta, k=5)

    if len(docs) == 0:
        st.write("Não encontrei essa informação nos documentos internos.")
        st.stop()

    contexto = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    Você é um assistente interno da Claro. Responda somente com base nos documentos abaixo:

    DOCUMENTOS:
    {contexto}

    PERGUNTA:
    {pergunta}

    Se a resposta não estiver nos documentos, diga que não consta.
    """

    resposta = llm.invoke(prompt)

    st.write("### Resposta")
    st.write(resposta.content)
