import streamlit as st
import subprocess
import sys
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# ---------------------------------------------------------
# TÍTULO DO APP
# ---------------------------------------------------------
st.title("Assistente Interno da Claro - Protótipo")
st.write("Pergunte sobre RH, TI ou documentos internos.")

# ---------------------------------------------------------
# BOTÃO: RODAR INGEST.PY E ATUALIZAR A BASE
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# MODELO OPENAI
# ---------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ---------------------------------------------------------
# EMBEDDINGS LOCAIS
# ---------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------------
# CHROMA VECTORSTORE (AGORA COM persist_directory)
# ---------------------------------------------------------
db = Chroma(
    collection_name="claro_base",
    embedding_function=embeddings,
    persist_directory="chroma"
)

# ---------------------------------------------------------
# FUNÇÃO: REESCRITA DE PERGUNTA
# ---------------------------------------------------------
def melhorar_pergunta(pergunta):
    prompt = f"""
    Reescreva a pergunta abaixo de forma objetiva para busca em documentos internos.
    Não altere o significado, apenas deixe mais direta.

    PERGUNTA ORIGINAL:
    {pergunta}
    """
    resposta = llm.invoke(prompt)
    return resposta.content.strip()


# ---------------------------------------------------------
# FUNÇÃO: BUSCA COM SCORE
# ---------------------------------------------------------
def buscar_documentos(pergunta):
    pergunta_reescrita = melhorar_pergunta(pergunta)

    docs_com_scores = db.similarity_search_with_score(
        pergunta_reescrita,
        k=5
    )

    # Score do Chroma: quanto MAIOR, mais relevante
    docs_filtrados = [doc for doc, score in docs_com_scores if score > 0.3]

    return docs_filtrados, pergunta_reescrita


# ---------------------------------------------------------
# INTERFACE
# ---------------------------------------------------------
pergunta = st.text_input("Digite sua pergunta:")

if pergunta:

    docs_list, pergunta_reescrita = buscar_documentos(pergunta)

    if len(docs_list) == 0:
        st.write("### Resposta")
        st.write("❌ Não encontrei essa informação nos documentos internos.")

        with st.expander("Pergunta reformulada automaticamente"):
            st.write(pergunta_reescrita)

        st.stop()

    contexto = "\n\n".join([d.page_content for d in docs_list])

    prompt_resposta = f"""
    Você é um assistente interno da Claro. Responda SOMENTE com base nos documentos abaixo:

    DOCUMENTOS:
    {contexto}

    PERGUNTA ORIGINAL:
    {pergunta}

    PERGUNTA REESCRITA:
    {pergunta_reescrita}

    Responda de forma objetiva e apenas com informações presentes nos documentos.
    """

    resposta = llm.invoke(prompt_resposta)

    st.write("### Resposta")
    st.write(resposta.content)

    with st.expander("Documentos utilizados"):
        st.write(f"🔁 Pergunta reescrita: **{pergunta_reescrita}**")
        for d in docs_list:
            st.write("-" * 50)
            st.write(d.page_content[:600] + "...")
