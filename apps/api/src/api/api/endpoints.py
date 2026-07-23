from fastapi import APIRouter, Request
from api.api.models import AgentRequest, AgentResponse, RAGUsedContext
from api.agents.retrieval_generation import rag_pipeline_wrapper
from qdrant_client import QdrantClient
from api.agents.graph import agent_wrapper
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

rag_router = APIRouter()

@rag_router.post("/")
def chat(request: Request, payload: AgentRequest) -> AgentResponse:
    result = agent_wrapper(payload.query, payload.thread_id)
    return AgentResponse(
        answer=result['answer'],
        used_context=[RAGUsedContext(**item) for item in result['used_context']]
    )

api_router = APIRouter()
api_router.include_router(rag_router, prefix="/agent", tags=["agent"])