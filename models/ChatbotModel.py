import sys
sys.path.append('/home/walaa-shaban/Documents/project/chatbot-llama3')
import bs4
# from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from src.embedding import embedding
from langchain.llms import Ollama
from langchain_core.prompts import PromptTemplate
import chromadb
from langchain.vectorstores import Chroma


class ChatbotModel :

    docs = ["https://lilianweng.github.io/posts/2023-06-23-agent/#task-decomposition",
            "/home/walaa-shaban/Documents/project/chatbot-llama3/input/Introduction to Machine Learning with Python.pdf",
            "/home/walaa-shaban/Documents/project/chatbot-llama3/input/thebook.pdf"]
    
    chunk_size = None
    embed_model = None
    similarity = None
    llm  = None
    retriver = None
    
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
        print(pages[0])
        return pages

    def split_docs(self):
        loader3 = PyPDFLoader("/home/walaa-shaban/Documents/project/chatbot-llama3/input/Introduction to Machine Learning with Python.pdf")
        pages3 = loader3.load_and_split()
        text_splitters = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=100, add_start_index=True)
        all_splits = text_splitters.split_documents(pages3)
        return all_splits
       

    def vector_db(self):
        vector_database = Chroma.from_documents(documents=self.split_docs(),embedding=embedding(self.embed_model))
        retriever = vector_database.as_retriever(search_type=self.similarity,search_kwargs={"k":4})
        return retriever

    def __init__(self, chunk_size:int, embed_model, similarity:str) -> None:
        self.chunk_size = chunk_size
        self.similarity = similarity
        self.embed_model = embed_model
        self.retriver = self.vector_db()
        self.llm = Ollama(model="llama3:latest")


    

