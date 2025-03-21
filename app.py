import os
import time
import streamlit as st
import fitz  # PyMuPDF
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint

huggingface_api_key = st.secrets["huggingface_api_key"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key

# Streamlit UI
st.title("📄 AccuFetch")
st.subheader("Find What Matters, Instantly!")
# Add a description under the subheader
st.markdown(
"""
    This tool allows you to **upload documents** and extract key information using AI-powered models. 

    **Please note:** The initial document reading and processing might take some time, especially for large documents. Once it's done, you can start asking questions to get detailed answers.

    - **Upload documents**: Use the sidebar to upload your document.
    - **Ask questions**: Type your queries in the text box.
    - **Click Submit button**
"""
)
st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])


# Cache the file reading, text splitting, and embedding creation
@st.cache_data
def read_and_process_file(uploaded_file):
    # Read the PDF from BytesIO
    pdf_data = uploaded_file.read()
    pdf_stream = BytesIO(pdf_data)  # Create a BytesIO stream from the uploaded file

    # Load PDF using fitz (PyMuPDF) and extract text
    doc = fitz.open(stream=pdf_stream)
    text = ""
    for page in doc:
        text += page.get_text()

    # Split text into optimized chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])

    # Convert to embeddings
    start_time = time.time()
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    doc_texts = [doc.page_content for doc in docs]
    doc_embeddings = embeddings.embed_documents(doc_texts)
    end_time = time.time()

    return doc_texts, doc_embeddings, embeddings


# Create FAISS vector store
@st.cache_data
def create_vectorstore(doc_embeddings, doc_texts, _embeddings):
    vectorstore = FAISS.from_texts(doc_texts, _embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    return retriever


# Load LLM model
@st.cache_data
def load_llm(huggingface_api_key):
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        token=huggingface_api_key,
        temperature=0.6,
    )
    return llm


# Streamlit logic
if uploaded_file is not None:
    st.write("Reading file...")
    start_time = time.time()

    # Call the caching functions
    doc_texts, doc_embeddings, embeddings = read_and_process_file(uploaded_file)

    end_time = time.time()
    st.write(f"✅ File processing completed in {end_time - start_time:.2f} seconds")

    # Create vector store
    st.write("Creating vector store...")
    start_time = time.time()

    retriever = create_vectorstore(doc_embeddings, doc_texts, embeddings)

    end_time = time.time()
    st.write(
        f"✅ Vector store creation completed in {end_time - start_time:.2f} seconds"
    )

    # Load LLM model
    st.write("Loading LLM model...")
    start_time = time.time()

    llm = load_llm(huggingface_api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    end_time = time.time()
    st.write(f"✅ LLM model loaded in {end_time - start_time:.2f} seconds")

    # User Input
    query = st.text_input("Ask a question about the document and click Submit button:")
    submit_button = st.button("Submit")

    if submit_button and query:
        st.write("Searching for the answer...")
        start_time = time.time()

        response = qa_chain.invoke({"query": query})

        end_time = time.time()
        st.write(f"✅ Answer retrieved in {end_time - start_time:.2f} seconds")
        st.write("### Answer:")
        st.write(response["result"])
