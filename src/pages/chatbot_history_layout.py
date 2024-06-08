import sys
sys.path.append('C:\\Users\\Walaa Shaaban\\Documents\\chatbot-llama3')
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

st.header("Chat with llama3 docs  💬 📚")

model = ChatbotModel(chunk_size=500, similarity="similarity", embed_model=SentenceTransformer('all-MiniLM-L6-v2'))
st.write(model.retriver)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = """
You are an AI bot developed , your role is to answer the users questions from the knowledge you know,
At the end of the answer thank the user.
The answer must be detailed and no less than 100 words.
if you are asked about yourself, tell the user that I am chatbot.

what to do if the answer is not encluded in the prompt or the context:
    1. apologies to the user.
    2. tell the user that you do not know the answer for the asked question.
    3. tell the user that you are specialized in the communication only.
    4. ask the user if he has more questions to ask.
    5. do not mention anything about the context.

knowledge you know:
{context}

Question: {question}

answer:
"""

custom_rag_prompt = PromptTemplate.from_template(template)

docs = model.vector_db()
formatted_context = format_docs(docs)
context_and_question = {"context": formatted_context, 'question': RunnablePassthrough()}

rag_chain = (
    context_and_question
    | custom_rag_prompt
    | model.llm
    | StrOutputParser()
)


def get_response():
    user_message = st.session_state.chat_text
    chunks = []
    for chunk in rag_chain.stream(st.session_state.chat_text):
        chunks.append(chunk)
    full_text = "".join(chunks)
    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.history.append({"message": full_text, "is_user": False})


if "history" not in st.session_state:
    st.session_state.history = []

st.text_input("Enter your question ...",key="chat_text", on_change=get_response)

for i, chat in enumerate(st.session_state.history):
    st_message(**chat, key=str(i)) 