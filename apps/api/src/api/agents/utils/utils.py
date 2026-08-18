from langchain_core.messages import AIMessage

def postprocess_response(response, final_response_class, agent_name=None):
    final_answer = False
    answer = ""
    references = []

    def _sanitize_response(response, prefix):
        for tool_call in response.tool_calls:
            answer = tool_call.get('args').get('answer')
        
        return AIMessage(content=f'{prefix}: {answer}')    

    if len(response.tool_calls) > 0:
        for tool_call in response.tool_calls:
            if tool_call.get('name') == final_response_class:
                final_answer = True
                answer = tool_call.get('args').get('answer')
                prefix = f'[{agent_name} final answer]' if agent_name else ''
                references.extend(tool_call.get('args').get('references',[]))
                response = _sanitize_response(response, prefix)
    return {
        "answer": answer,
        "references": references,
        "response": response,
        "final_answer": final_answer
    }