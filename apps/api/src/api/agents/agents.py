from langsmith import traceable, get_current_run_tree
from langchain_core.messages import SystemMessage, convert_to_openai_messages, AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from api.agents.utils.prompt_management import prompt_template_config
from api.agents.tools import get_formatted_item_context, get_formatted_reviews_context
import instructor
import jinja2

## Pydantic models
class RAGUsedContext(BaseModel):
    id: str = Field(description="The ID of the item used to answer the question")
    description: str = Field(description="The description of the item used to answer the question")

class FinalResponse(BaseModel):
    """Call this tool when the final answer is possible using available context"""
    answer: str = Field(description="The answer to the user's question")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question")

class IntentRouterResponse(BaseModel):
    question_relevant: bool
    answer: str = Field(description="An answer to the question if it's not relevant, saying that the question is not relevant to the products in stock")

## Agent Node
@traceable(
    name= "agent_node",
    run_type= "llm",
    metadata= {
        "ls_provider": "openai",
        "ls_model": "gpt-5.4-mini"
    }
)
def agent_node(state) -> dict:
    template = prompt_template_config('api/agents/prompts/qa_agent.yml', 'qa_agent')
    prompt = template.render()

    llm = ChatOpenAI(
        model="gpt-5.4-mini",
        reasoning_effort = "low",
        use_responses_api=True,
    )

    llm_with_tools = llm.bind_tools(
        [get_formatted_item_context, get_formatted_reviews_context, FinalResponse],
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
        current_run.metadata['usage_metadata'] = {
            'input_tokens': response.usage_metadata['input_tokens'],
            'output_tokens': response.usage_metadata['output_tokens'],
            'total_tokens': response.usage_metadata['total_tokens'],
        }

    final_answer = False
    answer = ""
    references = []

    def sanitize_response(response):
        for tool_call in response.tool_calls:
            if tool_call.get('name') == "FinalResponse":
                answer = tool_call.get('args').get('answer')
        
        return AIMessage(content=answer)    

    if len(response.tool_calls) > 0:
        for tool_call in response.tool_calls:
            if tool_call.get('name') == 'FinalResponse':
                final_answer = True
                answer = tool_call.get('args').get('answer')
                references.extend(tool_call.get('args').get('references'))
                response = sanitize_response(response)

    return {
        "messages": [response],
        "final_answer": final_answer,
        "iteration": state.iteration + 1,
        "answer": answer,
        "references": references
    }

## Intent Router Node
@traceable(
    name="route_intent",
    run_type="llm",
    metadata={
        "ls_provider": "openai",
        "ls_model": "gpt-5.4-mini"
    }
)
def intent_router_node(state) -> dict:
    template = prompt_template_config('api/agents/prompts/intent_router_agent.yml', 'intent_router_agent')
    prompt = template.render()

    messages = state.messages
    conversation = []
    conversation.append(convert_to_openai_messages(messages[-1]))

    client = instructor.from_provider(
        "openai/gpt-5.4-mini",
        mode=instructor.Mode.RESPONSES_TOOLS
    )

    response, raw_response = client.create_with_completion(
        messages = [
            {"role":"system", "content":prompt},
            *conversation
        ],
        response_model=IntentRouterResponse,
        reasoning={"effort":"none"}
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata['usage_metadata'] = {
            'input_tokens': raw_response.usage.input_tokens,
            'output_tokens': raw_response.usage.output_tokens,
            'total_tokens': raw_response.usage.total_tokens,
        }
        trace_id = str(current_run.trace_id)
    else:
        trace_id = ""
    return {
        'question_relevant': response.question_relevant,
        'answer': response.answer,
        'trace_id': trace_id
    }