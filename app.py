import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

st.title("Assistente Interno da Claro - Protótipo")
st.write("Pergunte sobre RH, TI ou documentos internos.")

# Modelo
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Carrega vetores
embeddings = OpenAIEmbeddings()
db = Chroma(collection_name="claro_base", embedding_function=embeddings)

# Caixa de texto
pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    # 1. Busca de documentos (compatível com TODAS versões)
    docs_list = db.similarity_search(pergunta, k=3)

    # 2. Monta contexto
    contexto = "\n\n".join([d.page_content for d in docs_list])

    prompt = f"""
    Você é um assistente interno da Claro. Use SOMENTE o contexto abaixo:

    CONTEXTO:
    {contexto}

    PERGUNTA:
    {pergunta}

    Responda de forma clara, objetiva e baseada nas políticas internas.
    """

    # 3. Gera resposta
    resposta = llm.invoke(prompt)

    st.write("### Resposta:")
    st.write(resposta.content)

    with st.expander("Documentos usados"):
        for d in docs_list:
            st.write(d.page_content[:400] + "...")

