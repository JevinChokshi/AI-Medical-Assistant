from fastapi import APIRouter, UploadFile, File
from typing import List
from modules.load_vectorstore import load_vectorstore
from fastapi.responses import JSONResponse
from logger import logger

router = APIRouter()

@router.post("/upload_pdfs/")

async def upload_pdfs(files: List[UploadFile] = File(...)):
    try:
        logger.info("Recieved Uploaded Files")
        load_vectorstore(files)
        logger.info(f"Document added to vectorstore")
        return {"message": "Files processed and vectorstore updated"}

    except Exception as e:
        logger.exception("Error during pdf upload")
        return JSONResponse(status_code=500, content={"error": str(e)})