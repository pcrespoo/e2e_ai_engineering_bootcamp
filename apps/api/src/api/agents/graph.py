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
def agent_wrapper(question: str, thread_id: str) -> dict:
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
            result = graph.invoke(initial_state, config=config)

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
        
    return {
        'answer': result.get('answer'),
        'used_context': used_context,
        'trace_id': result.get('trace_id',''),
    }