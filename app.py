import streamlit as st
import subprocess
import sys
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

st.title("Assistente Interno da Claro - Protótipo")
st.write("Pergunte sobre RH, TI ou documentos internos.")

# ---------------------------------------------------------
# BOTÃO: RODAR INGEST.PY
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
# EMBEDDINGS
# ---------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------------
# CHROMA PERSISTENTE
# ---------------------------------------------------------
db = Chroma(
    collection_name="claro_base",
    embedding_function=embeddings,
    persist_directory="chroma"      # <<< LENDO A PASTA CORRETA
)

)

# ---------------------------------------------------------
# QUERY REWRITING
# ---------------------------------------------------------
def melhorar_pergunta(pergunta):
    prompt = f"""
    Reescreva a pergunta abaixo de forma curta, objetiva e ideal para sistemas de busca:

    PERGUNTA ORIGINAL:
    {pergunta}
    """

    resposta = llm.invoke(prompt)
    return resposta.content

# ---------------------------------------------------------
# BUSCA INTELIGENTE
# ---------------------------------------------------------
def buscar_documentos(pergunta):
    pergunta_reescrita = melhorar_pergunta(pergunta)

    docs_scores = db.similarity_search_with_score(
        pergunta_reescrita,
        k=5
    )

    # filtra documentos irrelevantes
    docs_filtrados = [
        doc for doc, score in docs_scores if score < 0.75
    ]

    return docs_filtrados, pergunta_reescrita

# ---------------------------------------------------------
# INTERFACE
# ---------------------------------------------------------
pergunta = st.text_input("Digite sua pergunta:")

if pergunta:

    docs_list, pergunta_reescrita = buscar_documentos(pergunta)

    if len(docs_list) == 0:
        st.write("### Resposta")
        st.error("❌ Não encontrei essa informação nos documentos internos.")

        with st.expander("Pergunta reformulada automaticamente"):
            st.write(pergunta_reescrita)

        st.stop()

    contexto = "\n\n".join([d.page_content for d in docs_list])

    prompt_resposta = f"""
    Você é um assistente interno da Claro.
    Responda SOMENTE com base nos documentos abaixo — não invente nada.

    DOCUMENTOS ENCONTRADOS:
    {contexto}

    PERGUNTA ORIGINAL:
    {pergunta}

    PERGUNTA REESCRITA:
    {pergunta_reescrita}

    Responda de forma objetiva, clara e baseada no texto.
    """

    resposta = llm.invoke(prompt_resposta)

    st.write("### Resposta")
    st.write(resposta.content)

    with st.expander("Documentos utilizados"):
        st.write(f"Pergunta reescrita: **{pergunta_reescrita}**")
        for d in docs_list:
            st.write("-" * 40)
            st.write(d.page_content[:500] + "...")
