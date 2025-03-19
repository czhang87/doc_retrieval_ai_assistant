# AccuFetch: Find What Matters, Instantly!

## Overview

AccuFetch is an AI-powered document retrieval assistant that allows users to upload PDF files, extract text, and query the content using a large language model (LLM). It leverages FAISS for efficient vector storage, PyMuPDF (fitz) for fast PDF text extraction, and Hugging Face embeddings for semantic search.

## Features

- **Fast PDF Text Extraction**: Uses PyMuPDF (fitz) with multithreading for quick text extraction.
- **Text Chunking**: Splits extracted text into optimized segments for better retrieval performance.
- **Efficient Vector Search**: FAISS-based storage for quick document search and retrieval.
- **Semantic Search with LLM**: Uses Hugging Face models to process and respond to user queries based on document content.
- **Streamlit UI**: Simple and interactive interface for uploading PDFs and querying data.

## Installation

### Prerequisites

Ensure you have Python 3.8+ installed.

### Install Required Dependencies

Run the following command to install all required Python libraries:

```sh
pip install streamlit pymupdf langchain faiss-cpu sentence-transformers concurrent.futures
```

If you want to use GPU acceleration, install FAISS with GPU support:

```sh
pip install faiss-gpu
```

## Usage

1. **Run the application**:

```sh
streamlit run app.py
```

2. **Upload a PDF file** through the sidebar.
3. **Ask a question** about the document in the query input field.
4. **Receive an AI-generated answer** based on the document's content.

## Configuration

### Hugging Face API Key

Ensure you have a valid Hugging Face API key. Set it in `secret_api_keys.py`:

```python
huggingface_api_key = "your_huggingface_api_key"
```

## Optimizations & Performance Enhancements

- **Multithreaded PDF Processing**: Uses `concurrent.futures.ThreadPoolExecutor` to extract text faster.
- **Optimized Text Splitting**: Uses `RecursiveCharacterTextSplitter` for efficient chunking.
- **GPU Support**: Can be enabled for embeddings and FAISS indexing (if CUDA is available).

## Future Enhancements

- Support for additional document formats (DOCX, TXT, and link, etc.).
- Improved retrieval accuracy with hybrid search (BM25 + embeddings).
- Integration with cloud storage for document management.

## License

MIT License

