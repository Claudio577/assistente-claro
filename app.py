import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

st.title("Assistente Interno da Claro - Protótipo")
st.write("Pergunte sobre RH, TI ou documentos internos.")

# Modelo OpenAI (somente para resposta)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# EMBEDDINGS LOCAIS (sem OpenAI, sem limite)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Carregar vetores
db = Chroma(collection_name="claro_base", embedding_function=embeddings)

# Pergunta
pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    # Busca local, sem usar OpenAI
    docs_list = db.similarity_search(pergunta, k=3)

    contexto = "\n\n".join([d.page_content for d in docs_list])

    prompt = f"""
    Você é um assistente interno da Claro. Responda com base no seguinte contexto:

    {contexto}

    PERGUNTA: {pergunta}
    """

    resposta = llm.invoke(prompt)

    st.write("### Resposta:")
    st.write(resposta.content)

