import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

st.title("Assistente Interno da Claro - Protótipo")
st.write("Pergunte sobre RH, TI ou documentos internos.")

# Carrega o modelo
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Carrega vetores
embeddings = OpenAIEmbeddings()
db = Chroma(collection_name="claro_base", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Caixa de texto
pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    # 1. Buscar documentos relevantes
    docs = retriever.get_relevant_documents(pergunta)

    # 2. Juntar contexto
    contexto = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    Você é um assistente interno da Claro. Responda usando o contexto abaixo.
    Se não souber, diga que não encontrou na base de documentos.

    CONTEXTO:
    {contexto}

    PERGUNTA:
    {pergunta}

    Resposta:
    """

    # 3. Gerar resposta com LLM
    resposta = llm.invoke(prompt)

    st.write("### Resposta:")
    st.write(resposta.content)

    # Mostrar documentos recuperados (opcional)
    with st.expander("Ver documentos usados"):
        for d in docs:
            st.write(d.page_content[:500] + "...")
