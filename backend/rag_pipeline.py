# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import Chroma
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

class RAGPipeline:
    def __init__(self):
        """Initialize RAG chain once"""
        self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=self.embedding
        )
        
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        self.llm = ChatOllama(model="llama3", temperature=0)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True
        )
    
    def ask_question(self, query):
        """Ask a question using the pre-loaded chain"""
        result = self.qa_chain.invoke({"query": query})
        
        if isinstance(result, dict):
            answer = result.get("result", "No answer found.")
            sources = result.get("source_documents", [])
        else:
            answer = str(result)
            sources = []
        
        return answer, sources

# Create instance once at startup
rag_pipeline = RAGPipeline()












#1st version of RAG pipeline with chain loading on every query 
# # from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOllama
# from langchain.chains.retrieval_qa.base import RetrievalQA
# from langchain_community.vectorstores import Chroma
# # from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings

# def load_rag_chain():
#     embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#     vectorstore = Chroma(
#     persist_directory="chroma_db",
#     embedding_function=embedding
#     # client_settings={"anonymized_telemetry": "False"}
#     )

#     retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#     # llm = ChatOpenAI(temperature=0)
#     llm=ChatOllama(model="llama3", temperature=0)

#     qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     return_source_documents=True
#     )

#     return qa_chain

# # # Load chain ONCE at module startup
# # qa_chain = load_rag_chain()


# def ask_question(query):
#     qa_chain = load_rag_chain()
#     # result = qa_chain(query)
#     result = qa_chain.invoke({"query": query})
#     if isinstance(result, dict):
#         # answer = result["result"]
#         # sources = result["source_documents"]
#         answer = result.get("result", "No answer found.")
#         sources = result.get("source_documents", [])
#         for doc in sources:
#             print(doc.page_content,'********','/n')  # [:200] Print first 200 chars of each source document  
#     else:
#         answer = str(result)
#         sources = []
#     return answer, sources