import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint


huggingface_api_key = st.secrets["huggingface_api_key"]


def extract_text(page):
    return page.extract_text()


os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key

# Streamlit UI
st.title("📄 AccuFetch")
st.subheader("Find What Matters, Instantly!")
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF and extract text
    reader = PdfReader("temp.pdf")
    text = "\n".join(
        [page.extract_text() for page in reader.pages if page.extract_text()]
    )

    # Split text into optimized chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])

    # Convert to embeddings and store in FAISS
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10}
    )  # Retrieve more documents for better accuracy

    # Load LLM model
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        token=huggingface_api_key,
        temperature=0.6,
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # User Input
    query = st.text_input("Ask a question about the document:")
    submit_button = st.button("Submit")

    if submit_button and query:
        response = qa_chain.invoke({"query": query})
        st.write("### Answer:")
        st.write(response["result"])
