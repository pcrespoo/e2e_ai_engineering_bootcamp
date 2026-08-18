from pydantic import BaseModel
from typing import Annotated, List, Any
from operator import add
from api.agents.agents import RAGUsedContext, product_qna_agent, shopping_cart_agent, warehouse_manager_agent, coordinator_agent
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
from api.agents.tools import get_formatted_item_context, get_formatted_reviews_context, add_to_shopping_cart, remove_from_cart, get_shopping_cart, get_shopping_cart_for_see, check_warehouse_availability, reserve_warehouse_items
from langchain_core.messages import HumanMessage
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import json

class AgentProperties(BaseModel):
    final_answer: bool = False
    iteration: int = 0

class CoordinatorProperties(BaseModel):
    final_answer: bool = False
    iteration: int = 0
    next_agent: str = ""

class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    user_intent: str = ""
    product_qna_agent: AgentProperties = AgentProperties()
    shopping_cart_agent: AgentProperties = AgentProperties()
    coordinator_agent: CoordinatorProperties = CoordinatorProperties()
    warehouse_manager_agent: AgentProperties = AgentProperties()
    user_id: str = ""
    cart_id: str = ""
    answer: str = ""
    references: list[RAGUsedContext] = []
    trace_id: str = ""

## Routers
def product_qna_agent_tool_router(state: State) -> str:
    if state.product_qna_agent.final_answer:
        return "end"
    elif state.product_qna_agent.iteration > 5:
        return "end"
    elif len(state.messages[-1].tool_calls) > 0:
        return "tools"
    else:
        return "end"

def shopping_cart_agent_tool_router(state: State) -> str:
    if state.shopping_cart_agent.final_answer:
        return "end"
    elif state.shopping_cart_agent.iteration > 10:
        return "end"
    elif len(state.messages[-1].tool_calls) > 0:
        return "tools"
    else:
        return "end"

def warehouse_manager_agent_tool_router(state: State) -> str:
    if state.warehouse_manager_agent.final_answer:
        return "end"
    elif state.warehouse_manager_agent.iteration > 4:
        return "end"
    elif len(state.messages[-1].tool_calls) > 0:
        return "tools"
    else:
        return "end"

def coordinator_agent_edge(state: State) -> str:
    if state.coordinator_agent.final_answer:
        return "end"
    elif state.coordinator_agent.iteration > 6:
        return "end"
    elif state.coordinator_agent.next_agent == "product_qna_agent":
        return "product_qna_agent"
    elif state.coordinator_agent.next_agent == "shopping_cart_agent":
        return "shopping_cart_agent"
    elif state.coordinator_agent.next_agent == "warehouse_manager_agent":
        return "warehouse_manager_agent"
    else:
        return "end"
    
## Graph
workflow = StateGraph(State)

product_qna_tools = [get_formatted_item_context, get_formatted_reviews_context]
shopping_cart_tools = [add_to_shopping_cart, remove_from_cart, get_shopping_cart]
warehouse_manager_tools = [check_warehouse_availability, reserve_warehouse_items]

product_qna_tool_node = ToolNode(product_qna_tools)
shopping_cart_tool_node = ToolNode(shopping_cart_tools)
warehouse_manager_tool_node = ToolNode(warehouse_manager_tools)

workflow.add_node("product_qna_tool_node", product_qna_tool_node)
workflow.add_node("shopping_cart_tool_node", shopping_cart_tool_node)
workflow.add_node("warehouse_manager_tool_node", warehouse_manager_tool_node)
workflow.add_node("product_qna_agent", product_qna_agent)
workflow.add_node("shopping_cart_agent", shopping_cart_agent)
workflow.add_node("warehouse_manager_agent", warehouse_manager_agent)
workflow.add_node("coordinator_agent", coordinator_agent)

workflow.add_edge(START, "coordinator_agent")

workflow.add_conditional_edges(
    "coordinator_agent",
    coordinator_agent_edge,
    {
        "product_qna_agent": "product_qna_agent",
        "shopping_cart_agent": "shopping_cart_agent",
        "warehouse_manager_agent": "warehouse_manager_agent",
        "end": END 
    }
)

workflow.add_conditional_edges(
    "product_qna_agent",
    product_qna_agent_tool_router,
    {
        "tools": "product_qna_tool_node",
        "end": "coordinator_agent" 
    }
)

workflow.add_conditional_edges(
    "shopping_cart_agent",
    shopping_cart_agent_tool_router,
    {
        "tools": "shopping_cart_tool_node",
        "end": "coordinator_agent" 
    }
)

workflow.add_conditional_edges(
    "warehouse_manager_agent",
    warehouse_manager_agent_tool_router,
    {
        "tools": "warehouse_manager_tool_node",
        "end": "coordinator_agent" 
    }
)

workflow.add_edge("product_qna_tool_node", "product_qna_agent")
workflow.add_edge("shopping_cart_tool_node", "shopping_cart_agent")
workflow.add_edge("warehouse_manager_tool_node", "warehouse_manager_agent")

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
            elif tool_call.get("name") == "add_to_shopping_cart":
                return f"Adding {tool_call.get('args').get('items', [])} to the shopping cart."
            elif tool_call.get("name") == "remove_from_cart":
                return f"Removing {tool_call.get('args').get('product_id', '')} from the shopping cart."
            elif tool_call.get("name") == "get_shopping_cart":
                return f"Fetching the shopping cart..."
            elif tool_call.get("name") == "check_warehouse_availability":
                return f"Checking the warehouse availability..."
            elif tool_call.get("name") == "reserve_warehouse_items":
                return f"Reserving items in the warehouses..."

        if _is_node_start(chunk):
            if chunk[1].get("payload", {}).get("name") == "coordinator_agent":
                return "Analysing the question..."
            if chunk[1].get("payload", {}).get("name") == "product_qna_agent":
                return "Planning..."
            if chunk[1].get("payload", {}).get("name") == "shopping_cart_agent":
                return "Planning..."
            if chunk[1].get("payload", {}).get("name") == "warehouse_manager_agent":
                return "Planning..."
            if chunk[1].get("payload", {}).get("name").endswith("tool_node"):
                message = " ".join([_tool_to_text(tool_call) for tool_call in chunk[1].get('payload', {}).get('input', {}).messages[-1].tool_calls])
                return message

    qdrant_client = QdrantClient(url='http://qdrant:6333')

    initial_state = {
        "messages": [HumanMessage(content=question)],
        "user_id": thread_id,
        "cart_id": thread_id,
        "coordinator_agent": {
            "final_answer": False,
            "iteration": 0,
            "next_agent": ""
        },
        "product_qna_agent": {
            "final_answer": False,
            "iteration": 0
        },
        "shopping_cart_agent": {
            "final_answer": False,
            "iteration": 0
        },
        "warehouse_manager_agent": {
            "final_answer": False,
            "iteration": 0
        }
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
        
    shopping_cart = get_shopping_cart_for_see(user_id=thread_id, cart_id=thread_id)
    shopping_cart_items = [
        {
            'price': float(item.get('price')) if item.get('price') else None,
            'quantity': item.get('quantity'),
            'currency': item.get('currency'),
            'product_image_url': item.get('product_image_url'),
            'total_price': float(item.get('total_price')) if item.get('total_price') else None,
        }
        for item in shopping_cart
    ]
        
    yield _string_to_sse(json.dumps(
        {   
            'type': 'final_answer',
            'data': {   
                'answer': result.get('answer'),
                'used_context': used_context,
                'trace_id': result.get('trace_id',''),
                'shopping_cart': shopping_cart_items,
            }
        }
    ))