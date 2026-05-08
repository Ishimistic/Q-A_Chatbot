# RAG Chatbot API

A Retrieval-Augmented Generation (RAG) chatbot API built using FastAPI, LangChain, and HuggingFace models. This application loads a specific web document, creates vector embeddings for it, and uses a local language model to answer questions based on the retrieved context.

##  Features

- **Document Ingestion:** Automatically loads and processes web content (currently configured for [Lilian Weng's Agent post](https://lilianweng.github.io/posts/2023-06-23-agent/)).
- **Vector Search:** Uses `ChromaD`B and `sentence-transformers` for fast local document retrieval.
- **Local LLM:** Powered entirely by open-source HuggingFace models (`google/flan-t5-base`) running locally via pipelines. No paid external API keys required!
- **RESTful API:** Exposes endpoints via FastAPI for easy integration with frontends.

##  Tech Stack

- **Backend Framework:** FastAPI & Pydantic
- **RAG & Orchestration:** LangChain, LangChain Community
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Language Model:** HuggingFace `google/flan-t5-base`
- **Vector Store:** Chroma

##  Project Structure

```text
├── main.py             # FastAPI application entry point
├── routes.py           # API route definitions
├── rag_setup.py        # Core RAG logic (Embeddings, Vectorstore, LLM Chain)
├── requirements.txt    # Project dependencies
├── rag_setup.ipynb     # Jupyter Notebook for RAG experimentation/setup
└── .gitignore          # Git ignore rules for cached/environment files


## Installation
1. Clone the repository
```bash
git clone https://github.com/Ishimistic/Q-A_Chatbot.git
cd Q-A_Chatbot
```

2. Create virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
```
3. Install dependencies:
Make sure to install the project requirements along with fastapi and uvicorn.
```bash
pip install -r requirements.txt
pip install fastapi uvicorn pydantic
```

## Usage 
Start the FastAPI Server:

Run the following command in your terminal to start the server:
```bash
uvicorn main:app --reload
```

The API will start running at http://127.0.0.1:8000.

## API Endpoints
1. Root Endpoint (Check Status)
URL: /
Method: GET
Response:
```bash
{
  "message": "RAG API is running!"
}
```

2. Ask Question
URL: /ask
Method: POST
- Body:
```bash

{
  "query": "What is the main idea behind using LLMs as autonomous agents?"
}
```
- Response
```bash
{
  "question": "What is the main idea behind using LLMs as autonomous agents?",
  "answer": "[The model's generated answer based on the provided context]"
}
```

## Notes
First Run: The first time you start the application, it will download the HuggingFace models (flan-t5-base and all-MiniLM-L6-v2) to your local machine. This might take a few minutes depending on your internet connection speed.
