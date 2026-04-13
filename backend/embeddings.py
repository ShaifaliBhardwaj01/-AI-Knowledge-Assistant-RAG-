# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_store(chunks):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # vectorstore=Chroma(
    # persist_directory="chroma_db",
    # embedding_function=embedding
    # )
    vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="chroma_db"
    # client_settings={"anonymized_telemetry": "False"}
    )
    # if chunks:
    #     vectorstore.add_documents(chunks)   

    # vectorstore.persist()
    return vectorstore