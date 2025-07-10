# RAG Memory App with PDF Upload

A powerful Retrieval-Augmented Generation (RAG) application built with Streamlit that allows users to upload PDF documents and have intelligent conversations about their content using LangChain and Groq's LLaMA model.

## 🚀 Features

- **PDF Upload & Processing**: Upload single or multiple PDF files
- **Intelligent Document Chunking**: Automatically splits documents into manageable chunks
- **Vector Database**: Uses Chroma for efficient document retrieval
- **Conversational Memory**: Maintains chat history across sessions
- **Context-Aware Responses**: Understands conversation context and references
- **Session Management**: Multiple conversation sessions with unique IDs
- **Real-time Chat Interface**: Interactive Streamlit-based UI

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **LangChain**: Framework for building LLM applications
- **Groq**: Fast inference API for LLaMA models
- **Chroma**: Vector database for document storage
- **HuggingFace**: Embeddings model (BAAI/bge-small-en-v1.5)
- **PyPDF**: PDF document processing

## 📋 Prerequisites

- Python 3.8+
- Groq API key
- HuggingFace token (optional, for embeddings)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/prakhar175/RAG-chatbot.git
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   HF_TOKEN=your_huggingface_token_here
   ```

## 📦 Required Dependencies

Create a `requirements.txt` file with the following packages:

```txt
streamlit
python-dotenv
langchain
langchain-groq
langchain-community
langchain-huggingface
langchain-text-splitters
langchain-chroma
chromadb
pypdf
sentence-transformers
```

## 🚀 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**
   - Open your browser and go to `http://localhost:8501`

3. **Upload PDF documents**
   - Use the file uploader to select one or more PDF files
   - The app will automatically process and index the documents

4. **Start chatting**
   - Enter a session ID (or use the default)
   - Ask questions about the uploaded PDF content
   - The app maintains conversation history for context-aware responses

## 🎯 How It Works

1. **Document Processing**: 
   - PDFs are loaded and split into chunks using RecursiveCharacterTextSplitter
   - Chunks are embedded using HuggingFace's BAAI/bge-small-en-v1.5 model
   - Embeddings are stored in Chroma vector database

2. **Query Processing**:
   - User questions are contextualized using chat history
   - Relevant document chunks are retrieved using similarity search
   - LLaMA model generates responses based on retrieved context

3. **Memory Management**:
   - Each session maintains its own chat history
   - Conversation context is preserved across interactions

## 🔍 Key Components

### Vector Store Configuration
- **Embedding Model**: BAAI/bge-small-en-v1.5
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters

### LLM Configuration
- **Model**: meta-llama/llama-4-scout-17b-16e-instruct
- **Temperature**: 0.3 (balanced creativity/consistency)

### Retrieval Chain
- History-aware retriever for context understanding
- Combines document retrieval with conversational memory

## 📁 Project Structure

```
rag-memory-app/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── README.md            # This file
└── temp.pdf            # Temporary file (auto-generated when uploaded pdf)
```

## 🔑 API Keys Setup

### Groq API Key
1. Visit [Groq Console](https://console.groq.com)
2. Create an account and generate an API key
3. Add it to your `.env` file

### HuggingFace Token (Optional)
1. Visit [HuggingFace](https://huggingface.co)
2. Create an account and generate a token
3. Add it to your `.env` file

## 🎨 UI Features

- **File Upload**: Multiple PDF support with drag-and-drop
- **Session Management**: Unique session IDs for multiple conversations
- **Chat History**: Expandable section showing conversation history
- **Debug View**: Session state viewer for development
- **Real-time Responses**: Instant answers to user queries
