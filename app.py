import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

load_dotenv()

st.title("Assistente Interno da Claro - Protótipo")
st.write("Pergunte sobre RH, TI ou documentos internos.")

# Modelo
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Carregar vetores
embeddings = OpenAIEmbeddings()
db = Chroma(collection_name="claro_base", embedding_function=embeddings)

retriever = db.as_retriever(search_kwargs={"k": 3})

# Cadeia compatível com TODAS versões do LangChain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# Histórico necessário para a cadeia
if "history" not in st.session_state:
    st.session_state["history"] = []

# Interface
pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    resposta = qa_chain.run({
        "question": pergunta,
        "chat_history": st.session_state["history"]
    })
    
    st.session_state["history"].append((pergunta, resposta))

    st.write("### Resposta:")
    st.write(resposta)
