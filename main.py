import shutil
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rag_engine import RAGEngine


load_dotenv()

app = FastAPI(
    title="Research Paper Assistant",
    description="RAG-based Research Paper Assistant",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

rag = RAGEngine()

ALLOWED_EXTENSIONS = {".pdf"}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/api/documents")
async def get_documents():

    return {
        "documents": list(rag.document_names),
        "chunks": rag.chunk_count
    }


@app.post("/api/upload")
async def upload_papers(
    files: List[UploadFile] = File(...)
):

    if not files:

        return JSONResponse(
            status_code=400,
            content={
                "error": "No document uploaded."
            }
        )

    uploaded = []
    errors = []

    for file in files:

        filename = file.filename or ""

        extension = Path(filename).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:

            errors.append(
                f"{filename}: Unsupported file format. "
                f"Only PDF files are allowed."
            )

            continue

        safe_filename = Path(filename).name

        destination = UPLOAD_DIR / safe_filename

        try:

            with destination.open("wb") as buffer:

                shutil.copyfileobj(
                    file.file,
                    buffer
                )

            result = rag.add_pdf(destination)

            uploaded.append(result)

        except Exception as error:

            if destination.exists():
                destination.unlink(missing_ok=True)

            errors.append(
                f"{filename}: {str(error)}"
            )

    if not uploaded and errors:

        return JSONResponse(
            status_code=400,
            content={
                "errors": errors
            }
        )

    return {

        "message": "Documents processed successfully.",

        "uploaded": uploaded,

        "errors": errors,

        "documents": list(rag.document_names),

        "chunks": rag.chunk_count
    }


@app.post("/api/chat")
async def chat(payload: dict):

    question = str(
        payload.get("question", "")
    ).strip()

    if not question:

        return JSONResponse(
            status_code=400,
            content={
                "answer": "Please enter a question.",
                "sources": []
            }
        )

    if rag.chunk_count == 0:

        return JSONResponse(
            status_code=400,
            content={
                "answer":
                "No research paper has been uploaded yet. "
                "Please upload at least one PDF first.",

                "sources": []
            }
        )

    try:

        result = rag.ask(question)

        return result

    except Exception as error:

        return JSONResponse(
            status_code=500,
            content={
                "answer":
                f"An error occurred: {str(error)}",

                "sources": []
            }
        )


@app.delete("/api/documents")
async def clear_documents():

    rag.clear()

    for file in UPLOAD_DIR.glob("*"):

        if file.is_file():

            file.unlink(missing_ok=True)

    return {

        "message":
        "All uploaded documents and indexed content were cleared."
    }


@app.get("/health")
async def health():

    return {
        "status": "ok"
    }