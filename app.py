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

# ------------------ BOTÃO INGEST ------------------
if st.button("🔄 Atualizar base vetorial (rodar ingest)"):
    with st.spinner("Processando documentos e reconstruindo vetores..."):
        result = subprocess.run(
            [sys.executable, "ingest.py"],
            capture_output=True,
            text=True
        )
        st.success("Base vetorial atualizada com sucesso!")
        st.code(result.stdout)

st.write("---")

# ------------------ MODELO OPENAI ------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ------------------ EMBEDDINGS ------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------ CHROMA ------------------
db = Chroma(
    collection_name="claro_base",
    persist_directory="chroma",
    embedding_function=embeddings
)

# ------------------ REESCREVER PERGUNTA ------------------
def melhorar_pergunta(pergunta):
    prompt = f"""
    Reescreva a pergunta abaixo deixando ela mais objetiva para mecanismos de busca.
    Não mude o significado, só simplifique.

    PERGUNTA:
    {pergunta}
    """
    resposta = llm.invoke(prompt)
    return resposta.content.strip()

# ------------------ BUSCAR DOCUMENTOS ------------------
def buscar_documentos(pergunta):
    pergunta_reescrita = melhorar_pergunta(pergunta)

    resultados = db.similarity_search_with_score(
        pergunta_reescrita,
        k=5
    )

    docs_filtrados = [
        doc for doc, score in resultados if score < 0.65
    ]

    return docs_filtrados, pergunta_reescrita

# ------------------ INTERFACE ------------------
pergunta = st.text_input("Digite sua pergunta:")

if pergunta:

    docs, pergunta_reescrita = buscar_documentos(pergunta)

    if len(docs) == 0:
        st.write("### Resposta")
        st.error("❌ Não encontrei essa informação nos documentos internos.")

        with st.expander("Pergunta reformulada automaticamente"):
            st.write(pergunta_reescrita)

        st.stop()

    contexto = "\n\n".join([d.page_content for d in docs])

    prompt_final = f"""
    Você é um assistente interno da Claro. Responda SOMENTE com base no conteúdo abaixo.

    DOCUMENTOS:
    {contexto}

    PERGUNTA ORIGINAL:
    {pergunta}

    PERGUNTA REESCRITA:
    {pergunta_reescrita}

    Responda apenas se tiver certeza. 
    Se o conteúdo não aparecer nos documentos, diga claramente que não consta.
    """

    resposta = llm.invoke(prompt_final)

    st.write("### Resposta")
    st.write(resposta.content)

    with st.expander("Documentos utilizados"):
        st.write(f"Pergunta reescrita: **{pergunta_reescrita}**")
        for d in docs:
            st.write("-" * 40)
            st.write(d.page_content[:600] + "...")
