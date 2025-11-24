import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Carregar variáveis de ambiente
load_dotenv()

st.title("Assistente Interno da Claro - Protótipo")
st.write("Pergunte sobre RH, TI ou documentos internos.")

# Configurar o modelo
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Carregar a base vetorial
embeddings = OpenAIEmbeddings()
db = Chroma(collection_name="claro_base", embedding_function=embeddings)

# Criar cadeia RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever()
)

# Interface
pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    resposta = qa_chain.run(pergunta)
    st.write("### Resposta:")
    st.write(resposta)
