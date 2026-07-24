import openai
from qdrant_client.models import Prefetch, Document
from qdrant_client import models
import cohere

def get_embedding(text, model='text-embedding-3-small'):
    response = openai.embeddings.create(
        model=model,
        input=text,
    )
    
    return response.data[0].embedding

def retrieve_items_data(query, qdrant_client, collection_name='amazon-items-collection-01-hybrid-search', k=5):
    query_embedding = get_embedding(query)

    results = qdrant_client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=query_embedding,
                using="text-embedding-3-small",
                limit=20
            ),
            Prefetch(
                query=Document(
                    text=query,
                    model="qdrant/bm25",
                ),
                using="bm25",
                limit=20
            )
        ],
        query=models.RrfQuery(rrf=models.Rrf(weights=[3,1])),
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

def rerank_data(query, context, top_k=5):
    cohere_client = cohere.ClientV2()

    response = cohere_client.rerank(
        model='rerank-v4.0-pro',
        query=query,
        documents=context['retrieved_context_texts'],
        top_n=top_k
    )

    order = [result.index for result in response.results]

    return {
        'retrieved_context_ids': [context['retrieved_context_ids'][i] for i in order],
        'retrieved_context_texts': [context['retrieved_context_texts'][i] for i in order],
        'similarity_scores': [context['retrieved_context_scores'][i] for i in order],
        'retrieved_context_ratings': [context['retrieved_context_ratings'][i] for i in order]
    }
    
def process_context(retrieve_context):
    formatted_context = ''

    for id, chunk, rating in zip(retrieve_context['retrieved_context_ids'], retrieve_context['retrieved_context_texts'], retrieve_context['retrieved_context_ratings']):
        formatted_context += f"- Product ID: {id}, Product Rating: {rating}, Product Description: {chunk}\n"

    return formatted_context