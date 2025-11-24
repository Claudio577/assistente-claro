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

# Vetores
embeddings = OpenAIEmbeddings()
db = Chroma(collection_name="claro_base", embedding_function=embeddings)

# O retriever ANTIGO usa .query ou .invoke
retriever = db.as_retriever(search_kwargs={"k": 3})

# Caixa de texto
pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    # 1. Buscar documentos usando método universal
    try:
        docs = retriever.invoke(pergunta)
    except:
        docs = retriever.query(pergunta)

    # Se docs vier como lista direta
    if isinstance(docs, list):
        docs_list = docs
    else:
        # versões antigas retornam {"documents":[...]}
        docs_list = docs.get("documents", [])

    # 2. Montar contexto
    contexto = "\n\n".join([d.page_content for d in docs_list])

    prompt = f"""
    Você é um assistente interno da Claro. Use SOMENTE o contexto abaixo.

    CONTEXTO:
    {contexto}

    PERGUNTA:
    {pergunta}

    Responda de forma clara e objetiva.
    """

    # 3. Chamada ao LLM
    resposta = llm.invoke(prompt)

    st.write("### Resposta:")
    st.write(resposta.content)

    # Mostrar docs
    with st.expander("Documentos usados"):
        for d in docs_list:
            st.write(d.page_content[:500] + "...")
