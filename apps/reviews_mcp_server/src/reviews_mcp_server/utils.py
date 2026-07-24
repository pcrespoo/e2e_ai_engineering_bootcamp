import openai
from qdrant_client.models import Prefetch, Document, FusionQuery
from qdrant_client import models

def get_embedding(text, model='text-embedding-3-small'):
    response = openai.embeddings.create(
        model=model,
        input=text,
    )
    
    return response.data[0].embedding

def retrieve_prefiltered_reviews_data(query, qdrant_client, parent_asins, collection_name='amazon-reviews-collection-01', k=5):
    query_embedding = get_embedding(query)
    results = qdrant_client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=query_embedding,
                using="text-embedding-3-small",
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="parent_asin",
                            match=models.MatchAny(any=parent_asins)
                        )
                    ]
                ),
                limit=20
            )
        ],
        query=FusionQuery(fusion='rrf'),
        limit=k
    )

    retrieved_context_ids = []
    retrieved_context_texts = []
    similarity_scores = []

    for result in results.points:
        retrieved_context_ids.append(result.payload['parent_asin'])
        retrieved_context_texts.append(result.payload['preprocessed_data'])
        similarity_scores.append(result.score)
    return {
        'retrieved_context_ids': retrieved_context_ids,
        'retrieved_context_texts': retrieved_context_texts,
        'similarity_scores': similarity_scores
    }

def process_review_context(retrieve_context):
    formatted_context = ''

    for id, chunk in zip(retrieve_context['retrieved_context_ids'], retrieve_context['retrieved_context_texts']):
        formatted_context += f"- Product ID: {id}, Product Review: {chunk}\n"

    return formatted_context