# PDF Chatbot

✨ **Chat with Multiple PDF Files**: A Streamlit-powered application that lets users upload multiple PDF files and ask questions about their content. The application leverages advanced language models and vector embeddings to retrieve and process relevant information efficiently.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)

---

## Features

- **Multi-PDF Support**: Upload multiple PDF files simultaneously.
- **Contextual Question Answering**: Ask questions related to the uploaded PDFs and get precise answers.
- **Hybrid Retrieval System**:
  - Similarity-based retrieval using vector embeddings.
  - Multi-query generation for ambiguous questions.
- **Interactive UI**: Built with Streamlit for an intuitive user experience.

---

## Technologies Used

- **Python**: Core language for application logic.
- **Streamlit**: Interactive frontend for uploading files and displaying results.
- **LangChain**: For text splitting, prompt templates, and document retrieval.
- **Ollama**: Embedding and language model integration.
- **PyPDF2**: Extracting text from PDF files.
- **Chroma**: Vector database for embedding-based retrieval.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_folder>

2. **Install Dependencies: Ensure you have Python 3.9+ installed**.
   ```bash
   pip install -r requirements.txt

3. **Run the Application**:

   ```bash
   streamlit run app.py

---

## Project Structure

    ├── app.py                  # Main Streamlit application file
    ├── requirements.txt        # Project dependencies
    ├── readme.md               # Project documentation
    ├── src/
    │   ├── data/
    │   │   ├── embedded_vector.py  # Vector embedding and text extraction
    │   │   ├── retriever.py        # Document retrieval logic
    │   ├── model/
    │   │   ├── ollama_model.py     # Language model integration
    │   ├── prompt/
    │   │   ├── augment_prompt.py   # Augmenting prompts for better answers
    │   │   ├── retriever_prompt.py # Multi-query and agent prompt templates
    │   ├── utils/
    │   │   ├── config.py           # Configuration file for model settings
    │   │   ├── htmlTemplates.py    # HTML templates for responses

---

## How It Works

    1. Text Extraction: PDFs are parsed, and their content is extracted as plain text using PyPDF2.
    2. Text Chunking: Long texts are split into smaller, manageable chunks using LangChain.
    3. Vector Embedding: Text chunks are converted into vector embeddings via the Ollama embedding model.
    4. Hybrid Retrieval: Uses a combination of similarity-based and multi-query retrieval for precise document matching.
    5. Question Answering: The relevant documents are processed with the configured language model to generate an accurate answer.