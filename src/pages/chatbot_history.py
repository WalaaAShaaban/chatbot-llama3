import sys
sys.path.append('/home/walaa-shaban/Documents/project/chatbot-llama3/')
import streamlit as st
from streamlit_chat import message as st_message
from models.ChatbotModel import ChatbotModel 
from sentence_transformers import SentenceTransformer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# history
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

st.header("Chat with llama3 about documents 💬 📚")

model = ChatbotModel(chunk_size=500, similarity="similarity", embed_model=SentenceTransformer('all-MiniLM-L6-v2'))



if "history" not in st.session_state:
    st.session_state.history = []

contextualize_q_system_prompt = """
Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone \
question which can be understood without the chat history. Do not answer the question, just reformulate it if needed and otherwise return it as is.
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",contextualize_q_system_prompt),
        (MessagesPlaceholder("chat_history")),
        ("human", "{input}")
    ]
)

history_awar_retriever = create_history_aware_retriever(model.llm, model.retriver, contextualize_q_prompt)

qa_system_prompt = """
You are an assistant for question-answering tasks. use the following peices of retrieved context to answer the question.\
if you don't know the answer, just say that you don't know. Use four senteces maximum and keep the answer concise.\

{context}
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",qa_system_prompt), # role message pairs
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(model.llm,qa_prompt)
rag_chain = create_retrieval_chain(history_awar_retriever,question_answer_chain)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)


def get_response():
    user_message = st.session_state.chat_text
    response = conversational_rag_chain.invoke({"input":st.session_state.chat_text}, config={"configurable": {"session_id":"123"}})["answer"]
    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.history.append({"message": response, "is_user": False})

st.text_input("Enter your question ...",key="chat_text", on_change=get_response)

for i, chat in enumerate(st.session_state.history):
    st_message(**chat, key=str(i)) 