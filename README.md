# Private RAG (Retrieval-Augmented Generation) with Ollama

This project is a modified version of the example found in the Ollama repository, specifically from [ollama/examples/langchain-python-rag-privategpt](https://github.com/ollama/ollama/tree/main/examples/langchain-python-rag-privategpt).

## Description

This project implements a private Retrieval-Augmented Generation (RAG) system using Ollama, LangChain, and various document loaders. It allows you to ingest documents from different file formats, create embeddings, and then query your documents using a local language model, all without requiring an internet connection.

## Features

- Ingest documents from various file formats (PDF, TXT, DOCX, etc.)
- Create and store document embeddings locally
- Query your documents using a local language model
- Privacy-focused: all processing happens on your local machine

## Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/CHPCdoc-privateGPT.git
cd CHPCdoc-privateGPT
```
2. Install the required packages:
```
pip install -r requirements.txt
```
3. Make sure you have Ollama installed and running on your system. Please follow the instructions in the [CHPC Documentation](https://www.chpc.utah.edu/documentation/software/genai.php) regarding getting Ollama set up.
```
module load ollama
./ollama serve >& ollama.log
```


## Usage

1. Ingest documents:
```
python ingest.py
```
This script will process documents from the `source_documents` directory and create embeddings.

2. Query your documents:
```
python privateGPT.py
```
This will start an interactive session where you can ask questions about your documents.

## Environment Variables

- `PERSIST_DIRECTORY`: Directory to store the database (default: 'db')
- `SOURCE_DIRECTORY`: Directory containing the source documents (default: 'source_documents')
- `EMBEDDINGS_MODEL_NAME`: Name of the embeddings model to use (default: 'all-mpnet-base-v2')
- `MODEL`: Ollama model to use (default: 'llama3')

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgements

This project is based on the example from the [Ollama repository](https://github.com/ollama/ollama/tree/main/examples/langchain-python-rag-privategpt).
