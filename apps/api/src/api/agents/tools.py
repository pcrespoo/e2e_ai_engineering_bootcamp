from langsmith import traceable, get_current_run_tree
from langchain_core.tools import tool
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Document, FusionQuery
from qdrant_client import models
import openai
import cohere

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

## Items metadata retrieval tool
@traceable(
    name='retrieve_data',
    run_type='retriever',
)
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

@traceable(
    name='rerank_data',
    run_type='tool'
)
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
    
@traceable(
    name='format_retrieved_context',
    run_type='prompt',
)
def process_context(retrieve_context):
    formatted_context = ''

    for id, chunk, rating in zip(retrieve_context['retrieved_context_ids'], retrieve_context['retrieved_context_texts'], retrieve_context['retrieved_context_ratings']):
        formatted_context += f"- Product ID: {id}, Product Rating: {rating}, Product Description: {chunk}\n"

    return formatted_context

@tool
def get_formatted_item_context(query: str, top_k: int = 5) -> str:
    """
    Search available products and return the top k matching inventory items

    Expand the customer's question into 1-5 concise search statements and issue them in parallel in a single turn.
    Each statement covers one distinct product and attribute; statements should be independent and not express the same search intent.
    Use natural language for product description. If no brand or model is specified, search broadly rather than refusing.

    Example:
    "Earphones for me and a waterproof speaker" => "personal earphones", "waterproof speaker"
    "A warm winter jacket for hiking" => "insulated winter jacket", "hiking outerwear for cold weather"

    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more
    Returns:
        A string of the top_k context chunks with IDs and average ratings prepeding each chunk, each representing an inventory item for a given query
    """

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    retrieved_context = retrieve_items_data(
        query, 
        qdrant_client, 
        k=20
    )

    retrieved_context = rerank_data(query,retrieved_context,top_k=top_k)
    formatted_context = process_context(retrieved_context)

    return formatted_context

## Reviews retrieval tool
@traceable(
    name='retrieve_prefiltered_reviews_data',
    run_type='retriever',
)
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

@traceable(
    name='format_retrieved_reviews',
    run_type='prompt',
)
def process_review_context(retrieve_context):
    formatted_context = ''

    for id, chunk in zip(retrieve_context['retrieved_context_ids'], retrieve_context['retrieved_context_texts']):
        formatted_context += f"- Product ID: {id}, Product Review: {chunk}\n"

    return formatted_context

@tool
def get_formatted_reviews_context(query: str, parent_asins: list[str], top_k: int = 5) -> str:
    """
    Get the top_k reviews matching a query for a list of prefiltered items.
    Args:
        query: The query to get the top k reviews for
        parent_asins: The list of item IDs to prefilter for before running the query
        top_k: The number of reviews to retrieve, this should be at least 20 if multiple items are being filtered
    Returns:
        A string of the top_k context chunks with IDs and average ratings prepeding each chunk, each representing an inventory item for a given query
    """
    qdrant_client = QdrantClient(url="http://qdrant:6333")

    retrieved_context = retrieve_prefiltered_reviews_data(query, qdrant_client, parent_asins, collection_name='amazon-reviews-collection-01', k=top_k)

    formatted_context = process_review_context(retrieved_context)

    return formatted_context