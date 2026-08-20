import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from middlewares.exception_handlers import catch_exception_middeware

# 1. This function will load your routes in the background
async def initialize_rag_services():
    print("🚀 Port bound! Render checkpoint cleared successfully.")
    
    # Internal imports prevent Hugging Face/Pinecone from blocking the initial boot
    from routes.upload_pdfs import router as upload_router
    from routes.ask_question import router as ask_router
    
    app.include_router(upload_router)
    app.include_router(ask_router)
    
    print("✅ All routers, Pinecone connections, and Hugging Face models are ready!")

# 2. Define the non-blocking startup lifespan task
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This schedules the setup function in the background event loop
    asyncio.create_task(initialize_rag_services())
    yield

# 3. Create the app instance with the updated lifespan settings
app = FastAPI(
    title='MedicalAssistantAPI', 
    description="API for AI Medical Assistant Chatbot",
    lifespan=lifespan
)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Middleware exception handlers
app.middleware("http")(catch_exception_middeware)
