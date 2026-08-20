from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA 
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv


load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

class MedicalResponse(BaseModel):
    response: str = Field(description="The factual medical response derived strictly from context.")

def get_llm_chain(retriever):
    llm = ChatGroq(model="qwen/qwen3.6-27b")
    parser = PydanticOutputParser(pydantic_object=MedicalResponse)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template="""You are **MediBot**, an AI-powered assistant trained to help users understand medical documents and health-related questions.

        ### System Instructions:
        1. Provide a clear, accurate, and helpful response based **only on the provided context**.
        2. Respond in a calm, factual, and respectful tone.
        3. Use simple explanations when needed.
        4. If the context does not contain the answer, say exactly: "I'm sorry, but i could not find relevant information in the provided documents."
        5. DO NOT make up facts. Never use external knowledge outside the provided context.
        6. Do not give medical advice or a diagnosis.

        ---

        ### Context:
        {context}

        ---

        **User Question**: {question}

        **Answer**:"""
        )

    return RetrievalQA.from_chain_type(llm=llm, 
                                        chain_type="stuff", 
                                        retriever = retriever, 
                                        chain_type_kwargs={"prompt": prompt}, 
                                        return_source_documents=True)