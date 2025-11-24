import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

st.title("Assistente Interno da Claro - Protótipo")
st.write("Pergunte sobre RH, TI ou documentos internos.")

# Configurar o modelo
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Carregar base vetorial
embeddings = OpenAIEmbeddings()
db = Chroma(collection_name="claro_base", embedding_function=embeddings)

# Criar cadeia RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    chain_type="stuff"
)

# Interface
pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    resposta = qa_chain.run(pergunta)
    st.write("### Resposta:")
    st.write(resposta)
