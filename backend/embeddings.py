# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.openai import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def create_vector_store(chunks):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="./chroma_db"
    )

    vectorstore.persist()
    return vectorstore