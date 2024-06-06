import sys
sys.path.append('/home/walaa-shaban/Documents/project/chatbot-llama3')
import bs4
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from src.embedding import embedding
from langchain.llms import Ollama
from langchain_core.prompts import PromptTemplate


class ChatbotModel :

    docs = ["https://lilianweng.github.io/posts/2023-06-23-agent/#task-decomposition",
            "/home/walaa-shaban/Documents/project/chatbot-llama3/input/Introduction to Machine Learning with Python.pdf",
            "/home/walaa-shaban/Documents/project/chatbot-llama3/input/thebook.pdf"]
    
    chunk_size,  embed_model, similarity, llm  = None
    
    def input_doc(self):
        loader1 = WebBaseLoader(
        web_paths=(self.docs[0]),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(class_=("post-header","post-content","post-title"))
        )
        )
        page1 = loader1.load() 

        loader2 = PyPDFLoader(self.docs[1])
        pages2 = loader2.load_and_split()

        loader3 = PyPDFLoader(self.docs[2])
        pages3 = loader3.load_and_split()

        pages = page1 + pages2 + pages3

        return pages

    def split_docs(self):
        text_splitters = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=100, add_start_index=True)
        all_splits = text_splitters.split_documents(self.input_doc)
        return all_splits
       

    def vector_db(self):
        vector_database = Chroma.from_documents(documents=self.split_docs,embedding=embedding(self.embed_model))
        retriever = vector_database.as_retriever(search_type=self.similarity,search_kwargs={"k":4})
        return retriever

    def __init__(self, chunk_size:int, embed_model, similarity:str) -> None:
        self.chunk_size = chunk_size
        self.similarity = similarity
        self.embed_model = embed_model
        self.llm = Ollama(model="llama3:latest")

