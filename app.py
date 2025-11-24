import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

st.title("Assistente Interno da Claro - Protótipo")
st.write("Pergunte sobre RH, TI ou documentos internos.")

# Modelo da OpenAI (somente para responder)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Embeddings locais (sem limite, sem erro)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(collection_name="claro_base", embedding_function=embeddings)

pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    docs_list = db.similarity_search(pergunta, k=3)

    contexto = "\n\n".join([d.page_content for d in docs_list])

    prompt = f"""
    Você é um assistente interno da Claro. Responda SOMENTE com base no contexto abaixo:

    CONTEXTO:
    {contexto}

    PERGUNTA:
    {pergunta}

    Responda de modo claro, direto e correto.
    """

    resposta = llm.invoke(prompt)

    st.write("### Resposta")
    st.write(resposta.content)

    with st.expander("Documentos utilizados"):
        for d in docs_list:
            st.write(d.page_content[:400] + "...")
