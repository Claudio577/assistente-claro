import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

st.title("Assistente Interno da Claro - Protótipo")
st.write("Pergunte sobre RH, TI ou documentos internos.")

# ---------------------------
# MODELO PRINCIPAL (OpenAI)
# ---------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ---------------------------
# EMBEDDINGS LOCAIS
# ---------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------
# VETORSTORE DO CHROMA
# ---------------------------
db = Chroma(collection_name="claro_base", embedding_function=embeddings)

# ---------------------------
# REESCRITA DE PERGUNTA (Query Rewriting)
# ---------------------------
def melhorar_pergunta(pergunta):
    prompt = f"""
    Reescreva a pergunta abaixo de forma objetiva e otimizada para busca em documentos internos da Claro.
    Não mude o significado, apenas simplifique:

    Pergunta: {pergunta}
    """
    return llm.invoke(prompt).content

# ---------------------------
# BUSCA INTELIGENTE
# ---------------------------
def buscar_documentos(pergunta):
    # 1) Reescrever pergunta para melhorar a busca
    pergunta_melhorada = melhorar_pergunta(pergunta)

    # 2) Buscar documentos com pontuação
    docs_scores = db.similarity_search_with_score(pergunta_melhorada, k=5)

    # 3) Filtrar pela relevância
    docs_filtrados = [doc for doc, score in docs_scores if score < 0.65]

    return docs_filtrados, pergunta_melhorada


pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    docs_list, pergunta_reescrita = buscar_documentos(pergunta)

    # Se nada relevante foi encontrado
    if len(docs_list) == 0:
        st.write("### Resposta")
        st.write("Não encontrei essa informação nos documentos internos. Tente reformular a pergunta.")

        with st.expander("Pergunta reformulada automaticamente"):
            st.write(pergunta_reescrita)

        st.stop()

    # Montar contexto
    contexto = "\n\n".join([d.page_content for d in docs_list])

    # ---------------------------
    # PROMPT FINAL PARA RESPOSTA
    # ---------------------------
    prompt_final = f"""
    Você é um assistente interno da Claro. Responda com base SOMENTE nos documentos abaixo:

    DOCUMENTOS:
    {contexto}

    PERGUNTA DO USUÁRIO:
    {pergunta}

    PERGUNTA REESCRITA (para contexto):
    {pergunta_reescrita}

    Responda de forma clara, objetiva e correta, citando somente o que realmente aparece nos documentos.
    Se a informação não estiver nos documentos, diga que não consta.
    """

    resposta = llm.invoke(prompt_final).content

    # ---------------------------
    # EXIBIR RESULTADO
    # ---------------------------
    st.write("### Resposta")
    st.write(resposta)

    with st.expander("Documentos utilizados"):
        st.write(f"Pergunta reescrita: **{pergunta_reescrita}**\n")
        for d in docs_list:
            st.write("-" * 50)
            st.write(d.page_content[:600] + "...")

