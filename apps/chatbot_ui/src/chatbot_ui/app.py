from ast import Try
import streamlit as st
from chatbot_ui.core.config import config
import requests
import uuid

st.set_page_config(
    page_title="Ecommerce Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

thread_id = get_session_id()

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

if "messages" not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', 'content': 'How can I assist you today?'}]

if "used_context" not in st.session_state:
    st.session_state.used_context = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = thread_id

#Display the messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    suggestions_tab, = st.tabs(["Suggestions"])
    with suggestions_tab:
        if st.session_state.used_context:
            for idx, item in enumerate(st.session_state.used_context):
                st.caption(item.get('description', 'No Description'))
                if 'image_url' in item:
                    st.image(item['image_url'], width=250)
                st.caption(f"Price: {item['price']} USD")
                st.divider()
        else:
            st.info("No suggestions yet")

if prompt := st.chat_input("Hi, how can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        state, output = api_call('post', f'{config.API_URL}/agent', json={'query': prompt, 'thread_id': st.session_state.thread_id})
        answer = output['answer']
        used_context = output['used_context']
        st.session_state.used_context = used_context
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()


