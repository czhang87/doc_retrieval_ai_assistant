# AccuFetch

AccuFetch is a document retrieval assistant that uses AI to help you find relevant information from documents and webpages instantly. With AccuFetch, you can upload PDF, DOCX, or TXT files or simply provide a URL to extract content and search for answers to your queries.

## Features

- **File Upload Support**: Upload PDFs, DOCX, or TXT files for processing.
- **Web Scraping**: Input a URL, and AccuFetch will scrape the content of the webpage.
- **Text Extraction**: Automatically extracts text from various file types and web pages.
- **AI-Powered Search**: Uses a large language model (LLM) to retrieve answers based on the content of your document or webpage.
- **Instant Results**: Get accurate answers quickly from the uploaded files or websites.

## Installation

To run AccuFetch locally, clone this repository and install the required dependencies:

```bash
git clone https://github.com/czhang87/doc_retrieval_ai_assistant.git
cd doc_retrieval_ai_assistant
pip install -r requirements.txt

## API Key

AccuFetch uses the HuggingFace API for the large language model. You will need to provide an API key in the Streamlit secrets.

1. Go to [HuggingFace](https://huggingface.co) and sign up or log in.
2. Create an API key under your account settings.
3. Save the key in the `secrets.toml` file in your Streamlit app directory:

```toml
[huggingface_api_key]
huggingface_api_key = "your-api-key-here"
```

## Usage

1. Launch the Streamlit app:

```bash
streamlit run app.py
```

2. Open the app in your browser.
3. **Upload a Document**: You can upload a PDF, DOCX, or TXT file from your computer.
4. **Enter a Web Link**: If you have a URL to a webpage, you can paste it in the input field.
5. **Ask Questions**: After processing, you can ask questions about the content of the document or webpage.

### Example Queries:
- "What is the summary of the document?"
- "Find all mentions of financial terms."
- "What are the key points in the article?"

## File Processing Flow

1. **Extract Text**: The app extracts text from uploaded files or web pages.
2. **Chunk Text**: The text is split into smaller, manageable chunks for efficient processing.
3. **Generate Embeddings**: The app uses a HuggingFace model to generate embeddings for the text.
4. **Create Vector Store**: A FAISS index is created to enable fast retrieval of relevant information.
5. **Search**: Users can ask questions, and the AI retrieves the most relevant chunks of text to generate answers.

## Requirements

- `streamlit`
- `PyMuPDF` (for PDF processing)
- `python-docx` (for DOCX processing)
- `bs4` (for parsing HTML)
- `langchain`
- `langchain_community`
- `faiss-cpu`
- `huggingface_hub`
- `numpy`


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The application uses the [HuggingFace API](https://huggingface.co) for AI models.
- The project uses [Streamlit](https://streamlit.io) for building the user interface.
- [FAISS](https://github.com/facebookresearch/faiss) is used for fast similarity search.

For any issues or suggestions, feel free to open an issue or pull request in this repository.