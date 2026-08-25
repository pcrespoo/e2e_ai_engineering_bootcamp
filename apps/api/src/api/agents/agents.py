from langsmith import traceable, get_current_run_tree
from langchain_core.messages import SystemMessage, convert_to_openai_messages, AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List
from api.agents.utils.prompt_management import prompt_template_config
from api.agents.tools import get_formatted_item_context, get_formatted_reviews_context, add_to_shopping_cart, remove_from_cart, get_shopping_cart, check_warehouse_availability, reserve_warehouse_items
from api.agents.utils.utils import postprocess_response

## QnA Agent Response models
class RAGUsedContext(BaseModel):
    id: str = Field(description="The ID of the item used to answer the question")
    description: str = Field(description="The description of the item used to answer the question")

class FinalQnAAgentResponse(BaseModel):
    """Call this tool when the final answer is possible using available context"""
    answer: str = Field(description="The answer to the user's question")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question")

## General Agent Response model
class FinalAgentResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question")

## Coordinator Agent Response model
class Plan(BaseModel):
    next_agent: str = Field(description="The next agent to invoke")
    next_agent_task: str = Field(description="The task to be performed by the next agent")

## QnA Agent Node
@traceable(
    name= "product_qna_agent",
    run_type= "llm",
    metadata= {
        "ls_provider": "openai",
        "ls_model": "gpt-5.4-mini"
    }
)
def product_qna_agent(state) -> dict:
    template = prompt_template_config('api/agents/prompts/qa_agent.yml', 'qa_agent')
    prompt = template.render()

    llm = ChatOpenAI(
        model="gpt-5.4-mini",
        reasoning_effort = "low",
        use_responses_api=True,
    )

    llm_with_tools = llm.bind_tools(
        [get_formatted_item_context, get_formatted_reviews_context, FinalQnAAgentResponse],
        tool_choice="required"
    )

    response = llm_with_tools.invoke(
        [
            SystemMessage(content=prompt),
            *state.messages
        ]
    )

    postprocessed_response = postprocess_response(response, "FinalQnAAgentResponse", "product_qna_agent")
    final_answer = postprocessed_response.get("final_answer")
    answer = postprocessed_response.get("answer")
    references = postprocessed_response.get("references")
    response = postprocessed_response.get("response")

    return {
        "messages": [response],
        "product_qna_agent": {
            "final_answer": final_answer,
            "iteration": state.product_qna_agent.iteration + 1,
        },
        "answer": answer,
        "references": references
    }

@traceable(
    name= "shopping_cart_agent",
    run_type= "llm",
    metadata= {
        "ls_provider": "openai",
        "ls_model": "gpt-5.4-mini"
    }
)
def shopping_cart_agent(state) -> dict:
    template = prompt_template_config('api/agents/prompts/shopping_cart_agent.yml', 'shopping_cart_agent')
    prompt = template.render(user_id=state.user_id, cart_id=state.cart_id)

    llm = ChatOpenAI(
        model="gpt-5.4-mini",
        reasoning_effort = "low",
        use_responses_api=True,
    )

    llm_with_tools = llm.bind_tools(
        [add_to_shopping_cart, remove_from_cart, get_shopping_cart, FinalAgentResponse],
        tool_choice="required"
    )

    response = llm_with_tools.invoke(
        [
            SystemMessage(content=prompt),
            AIMessage(content=state.coordinator_agent.next_agent_task),
            *state.shopping_cart_agent.messages
        ]
    )

    postprocessed_response = postprocess_response(response, "FinalAgentResponse", "shopping_cart_agent")
    final_answer = postprocessed_response.get("final_answer")
    answer = postprocessed_response.get("answer")
    response = postprocessed_response.get("response")

    return {
        "messages": [response] if final_answer else [],
        "shopping_cart_agent": {
            "final_answer": final_answer,
            "iteration": state.shopping_cart_agent.iteration + 1,
            "messages": [response]
        },
        "answer": answer
    }

@traceable(
    name= "warehouse_manager_agent",
    run_type= "llm",
    metadata= {
        "ls_provider": "openai",
        "ls_model": "gpt-5.4-mini"
    }
)
def warehouse_manager_agent(state) -> dict:
    template = prompt_template_config('api/agents/prompts/warehouse_manager_agent.yml', 'warehouse_manager_agent')
    prompt = template.render()

    llm = ChatOpenAI(
        model="gpt-5.4-mini",
        reasoning_effort = "low",
        use_responses_api=True,
    )

    llm_with_tools = llm.bind_tools(
        [check_warehouse_availability, reserve_warehouse_items, FinalAgentResponse],
        tool_choice="required"
    )

    response = llm_with_tools.invoke(
        [
            SystemMessage(content=prompt),
            AIMessage(content=state.coordinator_agent.next_agent_task),
            *state.warehouse_manager_agent.messages
        ]
    )

    postprocessed_response = postprocess_response(response, "FinalAgentResponse", "warehouse_manager_agent")
    final_answer = postprocessed_response.get("final_answer")
    answer = postprocessed_response.get("answer")
    response = postprocessed_response.get("response")

    return {
        "messages": [response] if final_answer else [],
        "warehouse_manager_agent": {
            "final_answer": final_answer,
            "iteration": state.warehouse_manager_agent.iteration + 1,
            "messages": [response]
        },
        "answer": answer
    }

@traceable(
    name= "coordinator_agent",
    run_type= "llm",
    metadata= {
        "ls_provider": "openai",
        "ls_model": "gpt-5.4-mini"
    }
)
def coordinator_agent(state) -> dict:
    template = prompt_template_config('api/agents/prompts/coordinator_agent.yml', 'coordinator_agent')
    prompt = template.render()

    llm = ChatOpenAI(
        model="gpt-5.4-mini",
        reasoning_effort = "medium",
        use_responses_api=True,
    )

    llm_with_tools = llm.bind_tools(
        [FinalAgentResponse, Plan],
        tool_choice="required"
    )

    response = llm_with_tools.invoke(
        [
            SystemMessage(content=prompt),
            *state.messages #this lets the model reference the previous messages (the history)
        ]
    )

    current_run = get_current_run_tree()
    if current_run:
        trace_id = str(current_run.trace_id)
    else:
        trace_id = ''

    final_answer = False
    answer = ""
    next_agent = ""
    next_agent_task = ""

    if len(response.tool_calls) > 0:
        if response.tool_calls[0].get('name') == 'Plan':
            next_agent = response.tool_calls[0].get('args').get('next_agent')
            next_agent_task = response.tool_calls[0].get('args').get('next_agent_task')
            response = AIMessage(content=f'[coordinator_agent_decision]: Next Agent is {next_agent}. Next Task is {next_agent_task}')
        else:   
            postprocessed_response = postprocess_response(response, "FinalAgentResponse")
            final_answer = postprocessed_response.get("final_answer")
            answer = postprocessed_response.get("answer")
            response = postprocessed_response.get("response")

    return {
        "messages": [response] if response else [],
        "coordinator_agent": {
            "final_answer": final_answer,
            "iteration": state.coordinator_agent.iteration + 1,
            "next_agent": next_agent,
            "next_agent_task": next_agent_task
        },
        "answer": answer,
        "trace_id": trace_id
    }