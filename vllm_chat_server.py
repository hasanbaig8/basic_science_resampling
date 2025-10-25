"""
FastAPI server using vLLM Python API directly.

This provides an OpenAI-compatible chat completions endpoint.

Usage:
    python vllm_chat_server.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import uvicorn
import time
import uuid

# Configuration
MODEL_NAME = "Qwen/Qwen3-8b"
PORT = 8000

# Initialize FastAPI app
app = FastAPI(title="vLLM Chat Server", version="1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and tokenizer
llm = None
tokenizer = None


# Pydantic models for request/response
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]


@app.on_event("startup")
async def startup_event():
    """Load model and tokenizer on startup."""
    global llm, tokenizer

    print(f"Loading model: {MODEL_NAME}")
    print("This may take a few minutes...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load model with vLLM
    llm = LLM(
        model=MODEL_NAME,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        trust_remote_code=False
    )

    print(f"Model loaded successfully!")
    print(f"Server ready on http://0.0.0.0:{PORT}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "vLLM Chat Server",
        "model": MODEL_NAME,
        "endpoints": {
            "chat": "/v1/chat/completions",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    if llm is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": MODEL_NAME}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    """
    if llm is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert messages to prompt using chat template
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        prompt = tokenizer.apply_chat_template(
            messages_dict,
            tokenize=False,
            add_generation_prompt=True
        )

        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
        )

        # Generate response
        outputs = llm.generate([prompt], sampling_params)

        # Extract generated text
        generated_text = outputs[0].outputs[0].text

        # Create response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=generated_text),
                    finish_reason="stop"
                )
            ]
        )

        return response

    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print(f"Starting vLLM Chat Server on port {PORT}...")
    print(f"Model: {MODEL_NAME}")
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )
