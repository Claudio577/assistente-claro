# app.py
import streamlit as st
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# Configuração
st.title("Assistente Interno da Claro - Protótipo")
st.write("Pergunte sobre RH, TI ou documentos internos.")

# Carrega base vetorial
embeddings = OpenAIEmbeddings()
db = Chroma(collection_name="claro_base", embedding_function=embeddings)

# Cria cadeia RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3})
)

# Interface
pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    resposta = qa_chain.run(pergunta)
    st.write("### Resposta:")
    st.write(resposta)
