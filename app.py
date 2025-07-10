import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
load_dotenv()
os.environ["HF_TOKEN"]= os.getenv("HF_TOKEN")
groq=os.getenv("GROQ_API_KEY")
embeddings= HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
)

st.title("RAG Memory App with PDF uploadable")
st.write("Upload the PDF and chat regarding the content.")

llm = ChatGroq(
    groq_api_key=groq,
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.3
)
session_id=st.text_input("Enter your session ID", value="default_session")
if 'store' not in st.session_state:
    st.session_state.store = {}
uploaded_files=st.file_uploader("Choose the pdf file",type='pdf',accept_multiple_files=True)
if uploaded_files:
    documents=[]
    for uploaded_file in uploaded_files:
        temppdf=f"./temp.pdf"
        with open(temppdf,'wb') as file:
            file.write(uploaded_file.getvalue())
            file_name=uploaded_file.name
        loader=PyPDFLoader(temppdf)
        docs=loader.load()
        documents.extend(docs)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    splits= text_splitter.split_documents(documents)
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    retriever = vector_store.as_retriever()
    
    contextualize_q_prompt_system = (
    "Given a chat history and the user's latest question, "
    "which may reference previous messages, reformulate it into a standalone question "
    "that can be understood without the chat history. "
    "Do not answer the question. Only rewrite it if needed; otherwise, return it as is."
    )
    contextualize_q_prompt=ChatPromptTemplate.from_messages(    
        [
            ("system", contextualize_q_prompt_system),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt,
    )

    system_prompt = (
        "You are a helpful assistant that answers questions based on the provided context. "
        "Use the following pieces of context to answer the question. "
        "If you don't know the answer, just say that you don't know, don't try to make up an answer."
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt
    )
    rag_chain=create_retrieval_chain(
        history_aware_retriever,question_answer_chain)
    def get_session_history(session:str)-> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
    conversational_rag_chain=RunnableWithMessageHistory( 
        rag_chain,get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    user_input = st.text_input("Ask a question about the PDF content:")
    if user_input:
        session_history=get_session_history(session_id)
        response=conversational_rag_chain.invoke(
            {"input":user_input},
            config={
                "configurable":{"session_id":session_id}},
        )
        # Show assistant response
        st.write("Assistant:", response["answer"])

        with st.expander("View Chat History"):
            for msg in session_history.messages:
                if isinstance(msg, HumanMessage):
                    st.markdown(f"**User:** {msg.content}")
                elif isinstance(msg, AIMessage):
                    st.markdown(f"**Assistant:** {msg.content}")
                else:
                    st.markdown(f"**{type(msg).__name__}:** {msg.content}")
        with st.expander(" View session_state.store"):
            st.json({k: [m.content for m in v.messages] for k, v in st.session_state.store.items()})
