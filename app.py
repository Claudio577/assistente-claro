import streamlit as st
import subprocess
import sys
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

st.title("Assistente Interno da Claro - Protótipo")
st.write("Pergunte sobre RH, TI ou documentos internos.")

# Botão de ingest
if st.button("🔄 Atualizar base vetorial (rodar ingest)"):
    with st.spinner("Processando documentos e reconstruindo os vetores..."):
        result = subprocess.run(
            [sys.executable, "ingest.py"],
            capture_output=True,
            text=True
        )
        st.success("Base vetorial atualizada com sucesso!")
        st.code(result.stdout)

st.write("---")

# Modelo para resposta
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Carregar FAISS
try:
    db = FAISS.load_local("faiss_store", embeddings, allow_dangerous_deserialization=True)
except:
    db = None

def melhorar_pergunta(pergunta):
    prompt = f"""
    Reescreva esta pergunta de forma objetiva para busca em documentos internos:

    PERGUNTA:
    {pergunta}
    """

    return llm.invoke(prompt).content


pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    if db is None:
        st.error("❌ A base vetorial ainda não foi criada. Clique no botão acima para rodar o ingest.")
        st.stop()

    pergunta_reescrita = melhorar_pergunta(pergunta)
    docs = db.similarity_search(pergunta_reescrita, k=5)

    if len(docs) == 0:
        st.write("### Resposta")
        st.error("❌ Não encontrei essa informação nos documentos internos.")

        with st.expander("Pergunta reformulada automaticamente"):
            st.write(pergunta_reescrita)

        st.stop()

    contexto = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    Você é um assistente interno da Claro. Responda SOMENTE com base nos documentos abaixo.

    DOCUMENTOS ENCONTRADOS:
    {contexto}

    PERGUNTA ORIGINAL:
    {pergunta}

    PERGUNTA REESCRITA:
    {pergunta_reescrita}

    Responda somente com informações explícitas nos documentos.
    """

    resposta = llm.invoke(prompt)

    st.write("### Resposta")
    st.write(resposta.content)

    with st.expander("Documentos utilizados"):
        st.write(f"Pergunta reescrita: **{pergunta_reescrita}**")
        for d in docs:
            st.write("-" * 50)
            st.write(d.page_content[:500] + " ...")
