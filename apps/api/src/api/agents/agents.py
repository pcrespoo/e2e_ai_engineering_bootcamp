from openai import OpenAI
from google import genai
from groq import Groq
from api.core.config import config

def run_llm(provider, model_name, messages, max_tokens=500):
    if provider == "OpenAI":
        client = OpenAI(api_key=config.openai_api_key)
    elif provider == "Groq":
        client = Groq(api_key=config.groq_api_key)
    else:
        client = genai.Client(api_key=config.google_api_key)
    
    if provider == "Google":
        return client.models.generate_content(
            model=model_name,
            contents=[message['content'] for message in messages],
        ).text
    elif provider == "Groq":
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
        ).choices[0].message.content
    else:
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            reasoning='minimal',
        ).choices[0].message.content