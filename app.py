import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ---- Layout ----
st.title("Assistente Interno da Claro - Protótipo")
st.write("Pergunte sobre RH, TI ou documentos internos.")

# ---- Modelo de linguagem ----
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ---- Embeddings locais ----
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---- Banco vetorial ----
db = Chroma(
    collection_name="claro_base",
    embedding_function=embeddings
)

# ---- Entrada ----
pergunta = st.text_input("Digite sua pergunta:")

if pergunta:

    # EXPANDE a consulta para melhorar a busca semântica
    consulta_expandida = (
        f"{pergunta} beneficios vantagens direitos processo politica RH TI onboarding Claro"
    )

    # Busca documentos relevantes (mais tolerante)
    docs_list = db.similarity_search(consulta_expandida, k=5)

    # Monta contexto
    contexto = "\n\n".join([d.page_content for d in docs_list])

    # Prompt anti-alucinação
    prompt = f"""
Você é um assistente interno da Claro.
Responda APENAS com base no CONTEXTO abaixo.
Se a resposta não estiver no contexto, diga exatamente:
"Não encontrei essa informação nos documentos internos."

NÃO invente nada. NÃO use conhecimento externo.

-----------------------
CONTEXTO:
{contexto}
-----------------------

PERGUNTA:
{pergunta}

Responda de forma clara e objetiva.
    """

    # Gera resposta
    resposta = llm.invoke(prompt)

    # ---- Saída ----
    st.write("### Resposta")
    st.write(resposta.content)

    # Documentos usados
    with st.expander("Documentos utilizados"):
        for d in docs_list:
            st.write(d.page_content[:500] + "...")
