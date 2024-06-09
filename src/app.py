import streamlit as st
st.header("Chatbot Application  💬 📚")

st.write('''chatbot with history using Llama3
this task for create chatbot with history using llama3 and comparing the result when change:

1. chunk size
    - small chunk (500)
    - medium chunk (1000)
    - large chunk (2000)


2. embedding model
    - sentence_transformers
    - BERT
    - GPT

3. similarity type
    - cosine
    - dotProduct
''')