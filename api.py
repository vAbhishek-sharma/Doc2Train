### This is just a Proof of concept right Now  ###
###         Further research is needed         ###
# doc2train/api.py - REST API Interface
"""
FastAPI REST API for Doc2Train
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import tempfile
import uuid
import os
import shutil
from pathlib import Path
import asyncio
import uvicorn

# Import core functionality
from .core.pipeline import ProcessingPipeline
from .core.llm_client import get_available_providers, process_media_directly
from .utils.validation import validate_input_enhanced

app = FastAPI(
    title="Doc2Train API",
    description="Convert documents to AI training data via REST API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for job results
job_storage = {}

class ProcessingConfig(BaseModel):
    """Configuration for document processing"""
    mode: str = "extract_only"
    provider: Optional[str] = "openai"
    generators: List[str] = ["conversations", "qa_pairs"]
    output_format: str = "jsonl"
    chunk_size: int = 4000
    quality_threshold: float = 0.7
    use_cache: bool = True
    api_key: Optional[str] = None

class JobResponse(BaseModel):
    """Response for job submission"""
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    """Job status response"""
    job_id: str
    status: str
    progress: float
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Doc2Train API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": "2025-06-08T12:00:00Z"}

@app.get("/providers")
async def list_providers():
    """List available LLM providers"""
    try:
        providers = get_available_providers()
        return {"providers": providers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process", response_model=JobResponse)
async def process_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    config: ProcessingConfig = ProcessingConfig()
):
    """
    Process uploaded documents asynchronously
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Initialize job status
    job_storage[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "message": "Job queued for processing",
        "result": None,
        "error": None
    }

    # Start processing in background
    background_tasks.add_task(
        process_files_background,
        job_id,
        files,
        config
    )

    return JobResponse(
        job_id=job_id,
        status="queued",
        message=f"Processing {len(files)} files. Check status at /status/{job_id}"
    )

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get processing job status"""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")

    job_data = job_storage[job_id]
    return JobStatus(
        job_id=job_id,
        status=job_data["status"],
        progress=job_data["progress"],
        message=job_data["message"],
        result=job_data["result"],
        error=job_data["error"]
    )

@app.post("/analyze-media")
async def analyze_media_direct(
    file: UploadFile = File(...),
    provider: Optional[str] = None,
    prompt: Optional[str] = None,
    api_key: Optional[str] = None
):
    """
    Analyze media file directly with LLM (images, videos)
    """
    # Set API key if provided
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    try:
        # Process media directly
        result = process_media_directly(
            media_path=tmp_path,
            provider=provider,
            prompt=prompt
        )

        return {
            "filename": file.filename,
            "analysis": result,
            "provider": provider or "auto-detected"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass

@app.get("/download/{job_id}")
async def download_results(job_id: str):
    """Download processing results"""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")

    job_data = job_storage[job_id]
    if job_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")

    # Return result file if available
    result = job_data.get("result", {})
    output_file = result.get("output_file")

    if output_file and Path(output_file).exists():
        return FileResponse(
            output_file,
            media_type='application/octet-stream',
            filename=f"doc2train_results_{job_id}.jsonl"
        )
    else:
        return JSONResponse(content=result)

async def process_files_background(job_id: str, files: List[UploadFile], config: ProcessingConfig):
    """Background task for processing files"""
    temp_dir = None

    try:
        # Update status
        job_storage[job_id]["status"] = "processing"
        job_storage[job_id]["message"] = "Saving uploaded files..."
        job_storage[job_id]["progress"] = 0.1

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        file_paths = []

        # Save uploaded files
        for file in files:
            file_path = Path(temp_dir) / file.filename
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(file.file, f)
            file_paths.append(str(file_path))

        # Update status
        job_storage[job_id]["message"] = "Processing documents..."
        job_storage[job_id]["progress"] = 0.2

        # Set API key if provided
        if config.api_key:
            os.environ['OPENAI_API_KEY'] = config.api_key

        # Configure processing pipeline
        pipeline_config = {
            'mode': config.mode,
            'provider': config.provider,
            'generators': config.generators,
            'output_format': config.output_format,
            'chunk_size': config.chunk_size,
            'quality_threshold': config.quality_threshold,
            'use_cache': config.use_cache,
            'output_dir': temp_dir,
            'show_progress': False,  # API mode
            'verbose': False
        }

        # Process files
        pipeline = ProcessingPipeline(pipeline_config)
        result = pipeline.process_files(file_paths, None)

        # Update status - completed
        job_storage[job_id]["status"] = "completed"
        job_storage[job_id]["progress"] = 1.0
        job_storage[job_id]["message"] = "Processing completed successfully"
        job_storage[job_id]["result"] = result

    except Exception as e:
        # Update status - error
        job_storage[job_id]["status"] = "failed"
        job_storage[job_id]["message"] = "Processing failed"
        job_storage[job_id]["error"] = str(e)
        job_storage[job_id]["progress"] = 0.0

    finally:
        # Note: Don't clean up temp_dir immediately, user might want to download results
        pass

def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the API server"""
    print(f"""
🚀 Starting Doc2Train API Server

📡 API URL: http://{host}:{port}
📚 Documentation: http://{host}:{port}/docs
🔧 Admin Interface: http://{host}:{port}/redoc

Ready to process documents via API!
""")

    uvicorn.run("doc2train.api:app", host=host, port=port, reload=reload)

if __name__ == "__main__":
    start_server()

