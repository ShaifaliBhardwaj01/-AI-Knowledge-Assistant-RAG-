# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import Chroma
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_rag_chain():
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
    # client_settings={"anonymized_telemetry": "False"}
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # llm = ChatOpenAI(temperature=0)
    llm=ChatOllama(model="llama3", temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
    )

    return qa_chain


def ask_question(query):
    qa_chain = load_rag_chain()
    # result = qa_chain(query)
    result = qa_chain.invoke({"query": query})
    if isinstance(result, dict):
        # answer = result["result"]
        # sources = result["source_documents"]
        answer = result.get("result", "No answer found.")
        sources = result.get("source_documents", [])
    else:
        answer = str(result)
        sources = []
    return answer, sources