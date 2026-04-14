import streamlit as st
from backend.loader import load_and_split
from backend.embeddings import create_vector_store
# from backend.rag_pipeline import ask_question
from backend.rag_pipeline import rag_pipeline

st.title("📄 AI Knowledge Assistant (RAG)")

# Upload PDF
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with open("data/temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    if st.button("Process Document"):
        chunks = load_and_split("data/temp.pdf")
        create_vector_store(chunks)
        st.success("Document processed and stored!")

# Ask question
query = st.text_input("Ask a question from your document")

if query:
    answer, sources = rag_pipeline.ask_question(query)
    # answer, sources = ask_question(query)

    st.subheader("Answer:")
    st.write(answer)

    st.subheader("Sources:")
    for doc in sources:
        st.write(doc.page_content[:200])