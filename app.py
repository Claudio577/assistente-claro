import streamlit as st
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from ingest import processar_documentos

load_dotenv()

st.title("Assistente Interno da Claro - Protótipo")

# --- EXPLICAÇÃO DA TECNOLOGIA ---
st.markdown("""
### 🚀 O que é essa tecnologia?

Criamos um **Assistente Inteligente baseado em RAG (Retrieval-Augmented Generation)** — uma tecnologia que permite que a IA responda de forma precisa usando exclusivamente documentos internos da empresa.

Ela funciona assim:

- 📄 **Lê documentos internos** como PDFs, políticas, manuais e materiais de onboarding.  
- 🔍 **Transforma o conteúdo em vetores** por meio de embeddings (FAISS).  
- ❓ Quando uma pergunta é realizada:
  - O sistema busca automaticamente os trechos mais relevantes nos documentos.
  - Esses trechos são enviados como contexto para o modelo de IA.
- 🧠 A IA gera uma resposta clara baseada apenas no conteúdo disponível nos documentos da empresa.

Esse processo garante respostas rápidas, consistentes e alinhadas com as informações institucionais.
""")

st.write("Pergunte sobre RH, TI ou documentos internos.")

# ---- BOTÃO DE INGEST ----
if st.button("🔄 Atualizar base vetorial (rodar ingest)"):
    with st.spinner("Processando documentos e criando base vetorial..."):
        processar_documentos()
    st.success("Base vetorial criada com sucesso!")

# ---- VERIFICAR SE A BASE EXISTE ----
if not os.path.exists("base_faiss"):
    st.warning("⏳ Base vetorial não encontrada. Clique no botão acima para rodar o ingest.")
    st.stop()

# ---- CARREGAR A BASE ----
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("base_faiss", embeddings, allow_dangerous_deserialization=True)

# ---- OPENAI ----
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---- CAMPO DE PERGUNTA ----
pergunta = st.text_input("Digite sua pergunta:")

if pergunta:

    resultados = db.similarity_search(pergunta, k=5)

    if len(resultados) == 0:
        st.error("❌ Não encontrei essa informação nos documentos internos.")
        st.stop()

    contexto = "\n\n".join([doc.page_content for doc in resultados])

    prompt = f"""
    Responda APENAS com base nos documentos abaixo:

    {contexto}

    Pergunta: {pergunta}

    Se não houver resposta nos documentos, diga que não consta.
    """

    resposta = llm.invoke(prompt)
    st.write("### Resposta")
    st.write(resposta.content)

    with st.expander("Documentos utilizados"):
        for doc in resultados:
            st.write(doc.page_content[:500] + "...")
