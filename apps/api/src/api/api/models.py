from pydantic import BaseModel
from typing import Optional

class AgentRequest(BaseModel):
    query: str
    thread_id: str

class RAGUsedContext(BaseModel):
    image_url: str
    price: Optional[float] = None
    description: str

class AgentResponse(BaseModel):
    answer: str
    used_context: list[RAGUsedContext]