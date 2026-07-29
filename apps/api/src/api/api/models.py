from pydantic import BaseModel, Field
from typing import Optional, Union

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
    trace_id: str

class FeedbackRequest(BaseModel):
    trace_id: str
    feedback_score: Union[int, None] = Field(description="Feedback score, 0 or 1")
    feedback_text: str = Field(description="Feedback text")
    feedback_source_type: str = Field(description="Feedback source type, 'api' or 'user'")

class FeedbackResponse(BaseModel):
    message: str = Field(description="Message indicating the feedback was submitted successfully")