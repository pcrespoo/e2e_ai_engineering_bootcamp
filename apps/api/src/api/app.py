from fastapi import FastAPI, Request
from pydantic import BaseModel

from openai import OpenAI

from api.core.config import config

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def run_llm(provider, model_name, messages, max_tokens=500):
    ##TODO: Add other providers
    if provider == "OpenAI":
        client = OpenAI()
    
    ##TODO: add other API Calls instantion based on the provider
    return client.responses.create(
        model=model_name,
        input=messages,
        reasoning={'effort':'minimal'},
        max_output_tokens=max_tokens
    ).output_text

class ChatRequest(BaseModel):
    provider: str
    model_name: str
    messages: list[dict]

class ChatResponse(BaseModel):
    message: str

app = FastAPI()

@app.post("/chat")
def chat(
    request: Request,
    payload: ChatRequest,
) -> ChatResponse:

    response = run_llm(payload.provider, payload.model_name, payload.messages)

    return ChatResponse(message=response)