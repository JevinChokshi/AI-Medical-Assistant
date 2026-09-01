# AI Medical Assistant

A document-based medical question-answering application using **Retrieval-Augmented Generation (RAG)**. Users can upload one or more medical PDFs, ask questions about their contents, and receive answers generated from the relevant sections of the uploaded documents.

The application uses **Pinecone** for vector storage, **BAAI/bge-base-en-v1.5** for embeddings, **LangChain** for retrieval, and a **Groq-hosted Qwen model** for answer generation. The backend is exposed through APIs and the frontend is built with Streamlit.

## Live Demo

**Web App:**
https://medical-ai-assistant-by-jevin.streamlit.app/

**GitHub Repository:**
https://github.com/JevinChokshi/AI-Medical-Assistant

> **Note:** This application is intended for information retrieval from user-provided documents and is not a substitute for professional medical advice or diagnosis.

---

## Features

* Upload multiple PDF documents through the Streamlit interface.
* Extract and split PDF content into smaller chunks for retrieval.
* Generate vector embeddings using `BAAI/bge-base-en-v1.5`.
* Store document embeddings in Pinecone.
* Retrieve relevant document chunks for each user query.
* Generate answers using a Groq-hosted Qwen LLM.
* Use prompt engineering to control how retrieved context is used to generate answers.
* Custom `SimpleRetriever` implementation based on LangChain's `BaseRetriever`.
* Display retrieved source information alongside generated answers.
* Separate backend API and frontend application.
* Deployed backend on Render and frontend on Streamlit Cloud.

---

## Architecture

```text
                    ┌──────────────────────┐
                    │    Streamlit UI      │
                    │                      │
                    │ • Upload PDFs        │
                    │ • Ask Questions      │
                    │ • Display Answers    │
                    │ • Display Sources    │
                    └──────────┬───────────┘
                               │
                               │ HTTP Requests
                               ▼
                    ┌──────────────────────┐
                    │      FastAPI         │
                    │      Backend         │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐        ┌──────────────────┐
        │ PDF Processing  │        │  Question Answer │
        │                 │        │                  │
        │ • Extract text  │        │ • Query handling │
        │ • Chunk text    │        │ • Retrieval      │
        │ • Create        │        │ • Prompting      │
        │   embeddings    │        │ • LLM generation │
        └────────┬────────┘        └────────┬─────────┘
                 │                          │
                 ▼                          ▼
        ┌─────────────────┐        ┌──────────────────┐
        │ BGE Embeddings  │        │ LangChain        │
        │                 │        │ SimpleRetriever  │
        │ bge-base-en-v1.5│        └────────┬─────────┘
        └────────┬────────┘                 │
                 │                          ▼
                 └──────────────► ┌──────────────────┐
                                  │    Pinecone      │
                                  │  Vector Store    │
                                  └────────┬─────────┘
                                           │
                                           ▼
                                  ┌──────────────────┐
                                  │   Groq API       │
                                  │   Qwen LLM       │
                                  └──────────────────┘
```

---

## RAG Pipeline

The application follows a two-stage RAG workflow.

### 1. Document Ingestion

When PDFs are uploaded:

```text
PDF
 ↓
Text Extraction
 ↓
Text Chunking
 ↓
BGE Embeddings
 ↓
Pinecone Vector Store
```

The document content is divided into smaller chunks before generating embeddings. Each chunk is converted into a vector using:

```text
BAAI/bge-base-en-v1.5
```

The resulting vectors are stored in Pinecone for semantic retrieval.

### 2. Question Answering

When the user submits a question:

```text
User Question
      ↓
Query Embedding
      ↓
Pinecone Retrieval
      ↓
Relevant Chunks
      ↓
Prompt + Retrieved Context
      ↓
Qwen LLM via Groq
      ↓
Generated Answer
      ↓
Answer + Source Information
```

Only the retrieved document context is passed to the answer-generation stage, allowing the application to answer questions based on the uploaded PDFs rather than relying only on the model's general knowledge.

---

## Backend

The backend is implemented using **FastAPI** and provides two primary API routes.

### `POST /upload_pdfs`

Accepts one or more PDF files.

The route:

1. Receives uploaded PDFs.
2. Processes the document content.
3. Splits the content into chunks.
4. Generates embeddings using `BAAI/bge-base-en-v1.5`.
5. Stores the vectors in Pinecone.

### `POST /ask`

Accepts a user question and performs the RAG workflow.

The route:

1. Receives the question.
2. Retrieves relevant chunks from Pinecone.
3. Passes the retrieved context to the LangChain pipeline.
4. Applies the configured prompt.
5. Sends the prompt to the Qwen LLM through Groq.
6. Returns the generated answer and relevant source information.

---

## Retrieval

The project uses LangChain's `BaseRetriever` to implement a custom:

```python
SimpleRetriever
```

The retriever is responsible for connecting the question-answering pipeline with the vector store and returning the document chunks relevant to the user's query.

This separates retrieval from the LLM generation layer and makes the RAG pipeline easier to control and modify.

---

## Prompt Engineering

The application uses a structured prompt to guide the LLM's response using the retrieved document context.

The prompt is designed to:

* Keep the answer grounded in the retrieved context.
* Reduce unsupported responses.
* Provide relevant information from the uploaded documents.
* Separate the retrieved information from the model's general knowledge.
* Produce answers suitable for a document-based medical assistant.

The prompt is combined with the retrieved chunks before being sent to the Qwen model.

---

## Frontend

The frontend is built with **Streamlit**.

It provides:

### PDF Upload

Users can upload multiple PDFs through the interface.

### Chat Interface

Users can ask questions about the uploaded documents and receive answers through a conversational interface.

### Source Display

Along with the generated response, the application displays the source information associated with the retrieved content so users can identify where the answer came from.

---

## Project Structure

```text
AI-Medical-Assistant/
│
├── client/
│   ├── app.py
│   ├── config.py
│   ├── requirements.txt
│   │
│   ├── components/
│   │   ├── chatUI.py
│   │   ├── history_download.py
│   │   └── upload.py
│   │
│   └── utils/
│       └── api.py
│
├── server/
│   ├── main.py
│   ├── logger.py
│   ├── requirements.txt
│   │
│   ├── modules/
│   │   ├── llm.py
│   │   ├── load_vectorstore.py
│   │   ├── pdf.handlers.py
│   │   └── query_handlers.py
│   │
│   ├── routes/
│   │   ├── ask_question.py
│   │   └── upload_pdfs.py
│   │
│   └── middlewares/
│       └── exception_handlers.py
│
├── pyproject.toml
├── .python-version
└── README.md
```

The repository separates the Streamlit client from the API server, with dedicated modules for PDF processing, vector-store loading, LLM interaction, and query handling.

---

## Tech Stack

| Component           | Technology              |
| ------------------- | ----------------------- |
| Frontend            | Streamlit               |
| Backend             | FastAPI                 |
| RAG Framework       | LangChain               |
| Embedding Model     | `BAAI/bge-base-en-v1.5` |
| Vector Database     | Pinecone                |
| LLM Provider        | Groq                    |
| LLM                 | Qwen                    |
| API Communication   | HTTP/REST               |
| Backend Deployment  | Render                  |
| Frontend Deployment | Streamlit Cloud         |
| Language            | Python                  |

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/JevinChokshi/AI-Medical-Assistant.git

cd AI-Medical-Assistant
```

### 2. Set up the backend

```bash
cd server

pip install -r requirements.txt
```

Create the required environment variables:

```text
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

### 3. Set up the frontend

Open another terminal:

```bash
cd client

pip install -r requirements.txt
```

Configure the backend API URL in the client configuration.

Run Streamlit:

```bash
streamlit run app.py
```

The application can then be accessed through the local Streamlit URL.

---

## Deployment

### Backend — Render

The FastAPI backend is deployed on **Render**.

The backend exposes the API endpoints used by the Streamlit frontend:

```text
/upload_pdfs
/ask
```

Environment variables such as the Groq and Pinecone API keys are configured through the deployment environment rather than being stored in the repository.

### Frontend — Streamlit Cloud

The Streamlit frontend is deployed separately and communicates with the deployed backend through HTTP requests.

Live application:

https://medical-ai-assistant-by-jevin.streamlit.app/

---

## API Workflow

### Upload Documents

```text
Streamlit
    │
    │ POST /upload_pdfs
    ▼
FastAPI
    │
    ├── Extract PDF text
    ├── Split into chunks
    ├── Generate embeddings
    └── Store vectors
            │
            ▼
         Pinecone
```

### Ask a Question

```text
Streamlit
    │
    │ POST /ask
    ▼
FastAPI
    │
    ▼
SimpleRetriever
    │
    ▼
Pinecone
    │
    │ Relevant chunks
    ▼
Prompt + Context
    │
    ▼
Groq / Qwen
    │
    ▼
Answer + Sources
    │
    ▼
Streamlit Chat UI
```

---

## Example Use Case

A user can upload medical documents such as research papers, clinical reference documents, or educational material.

For example:

```text
Upload:
- diabetes.pdf
- treatment_guidelines.pdf
- research_paper.pdf
```

Then ask:

```text
What are the major risk factors discussed in the documents?
```

The system retrieves relevant sections from the uploaded PDFs and uses those sections as context for generating the answer.

---

## Limitations

* The quality of answers depends on the quality and content of the uploaded documents.
* Retrieval quality depends on chunking, embedding, and query similarity.
* The generated response should be treated as document-based information retrieval, not medical diagnosis or treatment advice.
* The application does not replace consultation with a qualified healthcare professional.

---

## Future Improvements

Possible improvements include:

* Add conversation-aware retrieval for multi-turn questions.
* Add metadata filtering for documents and pages.
* Add reranking after vector retrieval.
* Add evaluation metrics for retrieval and answer quality.
* Add document-level access controls.
* Add support for additional document formats.
* Add citation-level page and chunk references.
* Add automated RAG evaluation using benchmark question-answer pairs.

---

## Author

**Jevin Chokshi**

GitHub:
https://github.com/JevinChokshi
