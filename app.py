import os
import time
import streamlit as st
import fitz  # PyMuPDF
import docx
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint

# Load API key from Streamlit secrets
huggingface_api_key = st.secrets["huggingface_api_key"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key


# Function to extract text from different file types
def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "pdf":
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text = "\n".join([page.get_text() for page in doc])
    elif file_type == "docx":
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif file_type == "txt":
        text = uploaded_file.read().decode("utf-8")
    else:
        text = ""

    return text


# Function to extract text from a webpage
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return text
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch the URL: {e}")
        return ""


# Cache the file reading and processing
@st.cache_data
def read_and_process_file(uploaded_file=None, url=None):
    if uploaded_file:
        text = extract_text_from_file(uploaded_file)
    elif url:
        text = extract_text_from_url(url)
    else:
        text = ""

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])

    # Convert to embeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    doc_texts = [doc.page_content for doc in docs]
    doc_embeddings = embeddings.embed_documents(doc_texts)

    return doc_texts, doc_embeddings, embeddings


# Cache the vector store creation
@st.cache_data
def create_vectorstore(doc_texts, _embeddings):
    vectorstore = FAISS.from_texts(doc_texts, _embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    return retriever


# Cache the LLM model loading
@st.cache_data
def load_llm(huggingface_api_key):
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        token=huggingface_api_key,
        temperature=0.5,
    )
    return llm


# Streamlit UI
st.title("📄 AccuFetch")
st.subheader("Find What Matters, Instantly!")
st.info(
    "Upload a document or enter a web link at the sidebar to analyze its content. This may take some time for inital processing."
)

st.sidebar.header("Input Options")
uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF, DOCX, TXT file", type=["pdf", "docx", "txt"]
)
web_link = st.sidebar.text_input("Or enter a web link (URL):")

if uploaded_file or web_link:
    st.write("Reading and processing the document. Please wait...")
    start_time = time.time()

    # Process the file or URL
    doc_texts, doc_embeddings, embeddings = read_and_process_file(
        uploaded_file=uploaded_file, url=web_link
    )
    retriever = create_vectorstore(doc_texts, embeddings)
    llm = load_llm(huggingface_api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    end_time = time.time()
    st.write(f"✅ Processing completed in {end_time - start_time:.2f} seconds")

    # User Query Input
    query = st.text_input("Ask a question about the document or webpage:")
    submit_button = st.button("Submit")

    if submit_button and query:
        st.write("Searching for the answer...")

        system_prompt = (
            "You are an AI document retrieval assistant. Answer the question only based on "
            "the input document and eliminate hallucination. Don't show the question. Only show the answer."
        )
        combined_input = system_prompt + "\n" + query

        response = qa_chain.invoke({"query": combined_input})
        st.write("### Answer:")
        st.write(response["result"])
