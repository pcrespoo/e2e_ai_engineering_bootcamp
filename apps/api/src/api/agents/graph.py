from pydantic import BaseModel
from typing import Annotated, List, Any
from operator import add
from api.agents.agents import RAGUsedContext
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
from api.agents.tools import get_formatted_item_context, get_formatted_reviews_context
from api.agents.agents import agent_node, intent_router_node
from langchain_core.messages import HumanMessage
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from api.agents.retrieval_generation import rag_pipeline
import json

class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    question_relevant: bool = False
    iteration: int = 0
    answer: str = ""
    final_answer: bool = False
    references: list[RAGUsedContext] = []
    trace_id: str = ""

## Edges
def tool_router(state: State) -> str:
    if state.final_answer == True:
        return "end"
    elif state.iteration > 2:
        return "end"
    elif len(state.messages[-1].tool_calls) > 0:
        return "tools"
    else:
        return "end"

def intent_router_conditional_edges(state: State) -> str:
    if state.question_relevant:
        return "agent_node"
    else:
        return "end"
    
## Graph

workflow = StateGraph(State)
tools = [get_formatted_item_context, get_formatted_reviews_context]
tool_node = ToolNode(tools)

workflow.add_node("tool_node", tool_node)
workflow.add_node("agent_node", agent_node)
workflow.add_node("intent_router_node", intent_router_node)

workflow.add_edge(START, "intent_router_node")
workflow.add_conditional_edges(
    "intent_router_node",
    intent_router_conditional_edges,
    {
        "agent_node": "agent_node",
        "end": END 
    }
)
workflow.add_conditional_edges(
    "agent_node",
    tool_router,
    {
        "tools": "tool_node",
        "end": END 
    }
)

workflow.add_edge("tool_node", "agent_node")
graph = workflow.compile()

## Agent execution wrapper
def agent_stream_wrapper(question: str, thread_id: str) -> dict:
    def _string_to_sse(string: str):
        return f"data: {string}\n\n"

    def _process_graph_event(chunk):
        def _is_node_start(chunk):
            return chunk[1].get("type") == "task"

        def _tool_to_text(tool_call):
            if tool_call.get("name") == "get_formatted_item_context":
                return f"Looking for items: {tool_call.get('args').get('query', '')}."
            elif tool_call.get("name") == "get_formatted_reviews_context":
                return f"Fetching user reviews..."

        if _is_node_start(chunk):
            if chunk[1].get("payload", {}).get("name") == "intent_router_node":
                return "Analysing the question..."
            if chunk[1].get("payload", {}).get("name") == "agent_node":
                return "Planning..."
            if chunk[1].get("payload", {}).get("name") == "tool_node":
                message = " ".join([_tool_to_text(tool_call) for tool_call in chunk[1].get('payload', {}).get('input', {}).messages[-1].tool_calls])
                return message

    qdrant_client = QdrantClient(url='http://qdrant:6333')

    initial_state = {
        "messages": [HumanMessage(content=question)],
        "iteration": 0,
    }

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    with PostgresSaver.from_conn_string(
        "postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db"
        ) as checkpointer:
            graph = workflow.compile(checkpointer=checkpointer)
            for chunk in graph.stream(initial_state, config, stream_mode=["debug","values"]):
                processed_chunk = _process_graph_event(chunk)
                if processed_chunk: #only runs if the chunk corresponds to the beginning of a step
                    yield _string_to_sse(processed_chunk) #returns the progress to the frontend
                if chunk[0] == "values":
                    result = chunk[1] #returns the final result to the rest of the agent node execution

    used_context = []

    for reference in result.get('references', []):
        payload = qdrant_client.scroll(
            collection_name='amazon-items-collection-01-hybrid-search',
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key='parent_asin', match=MatchValue(value=reference.get('id')))
                ]
            ),
        )[0][0].payload

        image_url = payload.get('image', '')
        price = payload.get('price', None)
        
        if image_url:
            used_context.append({
                'image_url': image_url,
                'price': price,
                'description': reference.get('description'),
            })
        
    yield _string_to_sse(json.dumps(
        {   
            'type': 'final_answer',
            'data': {   
                'answer': result.get('answer'),
                'used_context': used_context,
                'trace_id': result.get('trace_id',''),
            }
        }
    ))