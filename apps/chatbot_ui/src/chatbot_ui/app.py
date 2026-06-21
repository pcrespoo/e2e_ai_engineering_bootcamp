from ast import Try
import streamlit as st
from chatbot_ui.core.config import config
import requests

def api_call(method, url, **kwargs):

    def _show_error_popup(message):
        st.session_state['error_popup'] = {
            "visible": True,
            "message": message,
        }
    
    try:
        response = getattr(requests, method)(url, **kwargs)
        
        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {'message':'Invalid response format from server'}
        
        if response.ok:
            return True, response_data
        
        return False, response_data
    except requests.exceptions.ConnectionError:
        _show_error_popup('Failed to connect to the server')
        return False, {'message':'Failed to connect to the server'}
    except requests.exceptions.Timeout:
        _show_error_popup('Request timed out')
        return False, {'message':'Request timed out'}
    except Exception as e:
        _show_error_popup(f'An error occurred: {e}')
        return False, {'message':f'An error occurred: {e}'}

with st.sidebar:
    # Dropdown for the provider and model name
    provider = st.selectbox("Provider", ["OpenAI"])
    model_name = st.selectbox("Model", ["gpt-5-nano", "gpt-5-mini"])

    # Store the provider and model name in the session state
    st.session_state.provider = provider
    st.session_state.model_name = model_name
    
if "messages" not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', 'content': 'How can I assist you today?'}]

#Display the messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hi, how can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = api_call('post', f'{config.API_URL}/chat', json={'provider': st.session_state.provider, 'model_name': st.session_state.model_name, 'messages': st.session_state.messages})
        answer = response[1]['message']
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})


