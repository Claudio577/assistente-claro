import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from ingest import processar_documentos  # usamos o ingest DIRETAMENTE

load_dotenv()

st.title("Assistente Interno da Claro - Protótipo")
st.write("Pergunte sobre RH, TI ou documentos internos.")

# ---------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------
INDEX_PATH = "faiss_index"

# ---------------------------------------------------------
# INICIALIZAÇÃO AUTOMÁTICA DA BASE VETORIAL
# ---------------------------------------------------------
@st.cache_resource
def carregar_base():
    """Carrega FAISS ou cria automaticamente se não existir."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not os.path.exists(INDEX_PATH):
        st.warning("⏳ Base vetorial não encontrada. Criando agora...")
        processar_documentos()
        st.success("✔ Base vetorial criada com sucesso!")

    # Carrega FAISS do disco
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# Carrega ou cria automaticamente
db = carregar_base()

# ---------------------------------------------------------
# MODELO DE LINGUAGEM
# ---------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ---------------------------------------------------------
# FUNÇÃO DE MELHORAR PERGUNTA
# ---------------------------------------------------------
def melhorar_pergunta(pergunta):
    prompt = f"""
    Reescreva a pergunta abaixo de forma mais direta e objetiva,
    pensando em otimizar a busca nos documentos internos da Claro.

    PERGUNTA ORIGINAL:
    {pergunta}
    """
    resposta = llm.invoke(prompt)
    return resposta.content

# ---------------------------------------------------------
# BUSCA VETORIAL
# ---------------------------------------------------------
def buscar(pergunta):
    pergunta_reescrita = melhorar_pergunta(pergunta)

    docs = db.similarity_search(
        pergunta_reescrita,
        k=5
    )

    return docs, pergunta_reescrita

# ---------------------------------------------------------
# INTERFACE
# ---------------------------------------------------------
pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    docs, pergunta_reescrita = buscar(pergunta)

    if len(docs) == 0:
        st.error("❌ Não encontrei essa informação nos documentos internos.")
        with st.expander("Pergunta reformulada automaticamente:"):
            st.write(pergunta_reescrita)
        st.stop()

    contexto = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
    Você é um assistente interno da Claro. Responda SOMENTE com base no conteúdo abaixo.

    DOCUMENTOS:
    {contexto}

    PERGUNTA ORIGINAL:
    {pergunta}

    PERGUNTA REESCRITA:
    {pergunta_reescrita}

    Se a resposta não estiver nos documentos, diga claramente: "Isso não consta nos documentos."
    """

    resposta = llm.invoke(prompt)

    st.write("### Resposta")
    st.write(resposta.content)

    with st.expander("Documentos utilizados"):
        st.write(f"Pergunta reescrita: **{pergunta_reescrita}**")
        for d in docs:
            st.write("-" * 40)
            st.write(d.page_content[:500] + "...")
