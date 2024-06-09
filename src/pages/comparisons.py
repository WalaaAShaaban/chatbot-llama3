import sys
sys.path.append('/home/walaa-shaban/Documents/project/chatbot-llama3/')
import streamlit as st
from sentence_transformers import SentenceTransformer
from models.ChatbotModel import ChatbotModel
from transformers import GPT2Model, BertModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

st.header("Comparisons result ❓")

col1, col2, col3 = st.columns(3)
with col1:
    chunk = st.selectbox(
"chunk size ?",
(500, 1000, 2000))
    
with col2:
    similarity = st.selectbox(
        "similarity type",
        ('similarity', 'mmr', "similarity_score_threshold")
    )
with col3:
    embedding_model = st.selectbox(
        'embedding type',
        ('Sentence Transformer', 'BERT', 'GPT')
    )
    if embedding_model == 'Sentence Transformer':
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    if embedding_model == 'GPT':
        model_name = 'gpt2'
        embed_model = GPT2Model.from_pretrained(model_name)
    if embedding_model == 'BERT':
        model_name = 'bert-base-uncased'
        embed_model = BertModel.from_pretrained(model_name)

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
model = ChatbotModel(chunk_size=chunk, similarity=similarity, embed_model=embed_model)
rag_chain = (
            {"context": model.retriver | format_docs , 'question': RunnablePassthrough()}
            | custom_rag_prompt
            | model.llm
            | StrOutputParser()
        )

question = st.text_input('input your question about documents...')
if question:
    st.write_stream(rag_chain.stream(question))