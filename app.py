import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import os

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

retriever = db.as_retriever(search_kwargs={"k": 3})

# Cadeia de RAG moderna e compatível
qa_chain = create_retrieval_chain(
    retriever=retriever,
    llm=llm
)

# Interface Streamlit
pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    response = qa_chain.invoke({"input": pergunta})
    st.write("### Resposta:")
    st.write(response["answer"])
