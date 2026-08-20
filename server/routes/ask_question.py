from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from pydantic import Field
from typing import List, Optional
from logger import logger
import os
import re

def clean_think_tags(text: str) -> str:
    """Removes leaked <think>...</think> reasoning tags from LLM outputs."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

router = APIRouter()

class SimpleRetriever(BaseRetriever):
            docs: List[Document]
            tags: Optional[List[str]] = Field(default_factory=list)
            metadata: Optional[dict] = Field(default_factory=dict)

            def _get_relevant_documents(self, query: str)-> List[Document]:
                return self.docs

@router.post("/ask/")
async def ask_question(question:str = Form(...)):
    try:
        logger.info(f"User Query: {question}")

        # embed model + pinecone setup
        pc=Pinecone(os.environ["PINECONE_API_KEY"])
        index=pc.index(os.environ["PINECONE_INDEX_NAME"])

        embed_model=HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True}) #GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        embedded_query=embed_model.embed_query(question)

        res=index.query(vector=embedded_query, top_k=3, include_metadata=True)

        docs = []
        sources = []

        for match in res['matches']:
            meta = match.get("metadata", {})
            # Checks 'text', 'page_content', or 'content' dynamically
            extracted_text = meta.get("text") or meta.get("page_content") or meta.get("content") or ""
            source_name = meta.get('source')

            if source_name:
                 sources.append(source_name)
            
            docs.append(
                Document(
                    page_content=extracted_text,
                    metadata=meta
                )
            )

        retriever = SimpleRetriever(docs=docs)
        chain = get_llm_chain(retriever)
        result = query_chain(chain, question)

        raw_response = result.get('result', result.get('response', ""))
        clean_response = clean_think_tags(str(raw_response))

        structured_output = {"response": clean_response, "sources": list(set(sources))}
        logger.info("Query is Successful")
        return JSONResponse(status_code=200, content=structured_output)
                
    except Exception as e:
        logger.exception("Error Processing Question")
        return JSONResponse(status_code=500, content={'error': str(e)})