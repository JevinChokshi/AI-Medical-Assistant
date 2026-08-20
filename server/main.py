from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from middlewares.exception_handlers import catch_exception_middeware

# 1. Use the Lifespan event to defer the slow initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
       
    # This imports your routes AFTER the app binds to the port
    from routes.upload_pdfs import router as upload_router
    from routes.ask_question import router as ask_router
    
    app.include_router(upload_router)
    app.include_router(ask_router)
    
    yield

# 2. Pass lifespan into FastAPI
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
