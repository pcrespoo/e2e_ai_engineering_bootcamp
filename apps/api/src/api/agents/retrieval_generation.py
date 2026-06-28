import openai
from qdrant_client import QdrantClient
from langsmith import traceable, get_current_run_tree

qdrant_client = QdrantClient(url='http://qdrant:6333')

@traceable(
    name='embed_query',
    run_type='embedding',
    metadata={
        'ls_provider': 'openai',
        'ls_model_name': 'text-embedding-3-small',
    },
)
def get_embedding(text, model='text-embedding-3-small'):
    response = openai.embeddings.create(
        model=model,
        input=text,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata['usage_metadata'] = {
            'input_tokens': response.usage.prompt_tokens,
            'total_tokens': response.usage.total_tokens,
        }
    
    return response.data[0].embedding

@traceable(
    name='retrieve_data',
    run_type='retriever',
)
def retrieve_data(query, qdrant_client, collection_name='amazon-items-collection-01', k=5):
    query_embedding = get_embedding(query)

    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=k
    )

    retrieved_context_ids = []
    retrieved_context_scores = []
    retrieved_context_texts = []
    retrieved_context_ratings = []

    for result in results.points:
        retrieved_context_ids.append(result.payload['parent_asin'])
        retrieved_context_scores.append(result.score)
        retrieved_context_texts.append(result.payload['processed_description'])
        retrieved_context_ratings.append(result.payload['average_rating'])

    return {
        'retrieved_context_ids': retrieved_context_ids,
        'retrieved_context_scores': retrieved_context_scores,
        'retrieved_context_texts': retrieved_context_texts,
        'retrieved_context_ratings': retrieved_context_ratings
    }

@traceable(
    name='format_retrieved_context',
    run_type='prompt',
)
def process_context(retrieve_context):
    formatted_context = ''

    for id, chunk, rating in zip(retrieve_context['retrieved_context_ids'], retrieve_context['retrieved_context_texts'], retrieve_context['retrieved_context_ratings']):
        formatted_context += f"- Product ID: {id}, Product Rating: {rating}, Product Description: {chunk}\n"

    return formatted_context

@traceable(
    name='build_prompt',
    run_type='prompt',
)
def build_prompt(question, formatted_context):
    prompt = f"""
    You are a shopping assistant that can answer questions about the products in stock.

    You will be given a question and a list of context.

    Instructions:
    - Answer the question based on the context only.
    - Never use word context and refer to it as the available products.
    - Do not use markdown formatting

    Context:
    {formatted_context}

    Question:
    {question}
    """
    return prompt

@traceable(
    name='generate_answer',
    run_type='llm',
    metadata={
        'ls_provider': 'openai',
        'ls_model_name': 'gpt-5.4-nano',
    },
)
def generate_answer(prompt):
    response = openai.chat.completions.create(
        model="gpt-5.4-nano",
        messages=[
            {"role": "system", "content": prompt}
        ],
        reasoning_effort='none'
    )
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata['usage_metadata'] = {
            'input_tokens': response.usage.prompt_tokens,
            'output_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens,
        }

    return response.choices[0].message.content

@traceable(
    name='rag_pipeline',
)
def rag_pipeline(question, qdrant_client, topk=5):
    retrieved_context = retrieve_data(query=question, qdrant_client=qdrant_client, k=topk)
    formatted_context = process_context(retrieved_context)
    prompt = build_prompt(question, formatted_context)
    answer = generate_answer(prompt)
    
    final_answer = {
        'answer': answer,
        'question': question,
        'retrieved_context_ids': retrieved_context['retrieved_context_ids'],
        'retrieved_context': retrieved_context['retrieved_context_texts'],
    }

    return final_answer