from langsmith import traceable, get_current_run_tree
from langchain_core.tools import tool
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Document, FusionQuery, Filter, FieldCondition, MatchValue, MatchAny
from qdrant_client import models
import openai
import cohere
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

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

    qdrant_client = QdrantClient(url="http://localhost:6333")

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
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_asin",
                            match=MatchAny(any=parent_asins)
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
    qdrant_client = QdrantClient(url="http://localhost:6333")

    retrieved_context = retrieve_prefiltered_reviews_data(query, qdrant_client, parent_asins, collection_name='amazon-reviews-collection-01', k=top_k)

    formatted_context = process_review_context(retrieved_context)

    return formatted_context

# Shopping Cart Tools
@tool
def add_to_shopping_cart(items: list[dict], user_id: str, cart_id: str) -> str:

    """Add a list of provided items to the shopping cart.
    
    Args:
        items: A list of items to add to the shopping cart. Each item is a dictionary with the following keys: product_id, quantity.
        user_id: The id of the user to add the items to the shopping cart.
        cart_id: The id of the shopping cart to add the items to.
        
    Returns:
        A list of the items added to the shopping cart.
    """

    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="tools_database",
        user="tools_user",
        password="tools_user_password"
    )
    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        
        for item in items:
            product_id = item['product_id']
            quantity = item['quantity']

            qdrant_client = QdrantClient(url="http://localhost:6333")

            dummy_vector = np.zeros(1536).tolist()
            payload = qdrant_client.query_points(
                collection_name="amazon-items-collection-01-hybrid-search",
                prefetch=[
                    Prefetch(
                        query=dummy_vector,
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key="parent_asin",
                                    match=MatchValue(value=product_id)
                                )
                            ]
                    ),
                        using="text-embedding-3-small",
                        limit=20
                    )
                ],
                query=FusionQuery(fusion="rrf"),
                limit=1,
            ).points[0].payload

            product_image_url = payload.get("image")
            price = payload.get("price")
            currency = 'USD'
        
            # Check if item already exists
            check_query = """
                SELECT id, quantity, price 
                FROM shopping_carts.shopping_cart_items 
                WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
            """
            cursor.execute(check_query, (user_id, cart_id, product_id))
            existing_item = cursor.fetchone()
            
            if existing_item:
                # Update existing item
                new_quantity = existing_item['quantity'] + quantity
                
                update_query = """
                    UPDATE shopping_carts.shopping_cart_items 
                    SET 
                        quantity = %s,
                        price = %s,
                        currency = %s,
                        product_image_url = COALESCE(%s, product_image_url)
                    WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
                    RETURNING id, quantity, price
                """
                
                cursor.execute(update_query, (new_quantity, price, currency, product_image_url, user_id, cart_id, product_id))
            
            else:
                # Insert new item
                insert_query = """
                    INSERT INTO shopping_carts.shopping_cart_items (
                        user_id, shopping_cart_id, product_id,
                        price, quantity, currency, product_image_url
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, quantity, price
                """
                
                cursor.execute(insert_query, (user_id, cart_id, product_id, price, quantity, currency, product_image_url))
            
    return f"Added {items} to the shopping cart."

@tool
def get_shopping_cart(user_id: str, cart_id: str) -> list[dict]:

    """
    Retrieve all items in a user's shopping cart.
    
    Args:
        user_id: User identifier
        cart_id: Cart identifier
    
    Returns:
        List of dictionaries containing cart items
    """
    
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="tools_database",
        user="tools_user",
        password="tools_user_password"
    )
    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:

        query = """
                SELECT 
                    product_id, price, quantity,
                    currency, product_image_url,
                    (price * quantity) as total_price
                FROM shopping_carts.shopping_cart_items 
                WHERE user_id = %s AND shopping_cart_id = %s
                ORDER BY added_at DESC
            """
        cursor.execute(query, (user_id, cart_id))

        return [dict(row) for row in cursor.fetchall()]

@tool
def remove_from_cart(product_id: str, user_id: str, cart_id: str) -> str:

    """
    Remove an item completely from the shopping cart.
    
    Args:
        user_id: User identifier
        product_id: Product identifier to remove
        cart_id: Cart identifier
    
    Returns:
        Information about the removal of the item from the shopping cart.
    """
    
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="tools_database",
        user="tools_user",
        password="tools_user_password"
    )
    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:

        query = """
                DELETE FROM shopping_carts.shopping_cart_items
                WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
            """
        cursor.execute(query, (user_id, cart_id, product_id))

        return f"Removed {product_id} from the shopping cart." if cursor.rowcount > 0 else f"Item {product_id} not found in the shopping cart."

# Warehouse Manager Tools
@tool
def check_warehouse_availability(items: list[dict]) -> dict:

    """Check availability of items across warehouses, including partial fulfillment options.
    
    Args:
        items: A list of items to check. Each item is a dictionary with keys: product_id, quantity.
        
    Returns:
        A dictionary containing:
        - can_fulfill_completely: bool indicating if all items can be fulfilled from at least one warehouse
        - warehouses_full_fulfillment: list of warehouses that can fulfill the entire order
        - warehouses_partial_fulfillment: list of warehouses with partial availability
        - unavailable_items: list of items that cannot be fulfilled from any warehouse
        - details: detailed breakdown per warehouse with availability for each item
    """
    
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="tools_database",
        user="tools_user",
        password="tools_user_password"
    )
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            result = {
                "can_fulfill_completely": False,
                "warehouses_full_fulfillment": [],
                "warehouses_partial_fulfillment": [],
                "unavailable_items": [],
                "details": []
            }
            
            # Check each warehouse for availability
            warehouse_query = """
                SELECT DISTINCT warehouse_id, warehouse_name, warehouse_location
                FROM warehouses.inventory
            """
            cursor.execute(warehouse_query)
            warehouses = cursor.fetchall()
            
            for warehouse in warehouses:
                warehouse_can_fulfill_all = True
                has_any_availability = False
                warehouse_details = {
                    "warehouse_id": warehouse['warehouse_id'],
                    "warehouse_name": warehouse['warehouse_name'],
                    "warehouse_location": warehouse['warehouse_location'],
                    "items": [],
                    "can_fulfill_all": False,
                    "has_partial": False
                }
                
                for item in items:
                    product_id = item['product_id']
                    requested_quantity = item['quantity']
                    
                    # Check availability in this warehouse
                    availability_query = """
                        SELECT product_id, total_quantity, reserved_quantity, available_quantity
                        FROM warehouses.inventory
                        WHERE warehouse_id = %s AND product_id = %s
                    """
                    cursor.execute(availability_query, (warehouse['warehouse_id'], product_id))
                    inventory = cursor.fetchone()
                    
                    available_qty = inventory['available_quantity'] if inventory else 0
                    
                    item_detail = {
                        "product_id": product_id,
                        "requested": requested_quantity,
                        "available": available_qty,
                        "can_fulfill_completely": available_qty >= requested_quantity,
                        "can_fulfill_partially": available_qty > 0 and available_qty < requested_quantity
                    }
                    
                    warehouse_details["items"].append(item_detail)
                    
                    # Track if warehouse can fulfill this item completely
                    if available_qty < requested_quantity:
                        warehouse_can_fulfill_all = False
                    
                    # Track if warehouse has any availability for any item
                    if available_qty > 0:
                        has_any_availability = True
                
                # Categorize warehouse
                if warehouse_can_fulfill_all:
                    warehouse_details["can_fulfill_all"] = True
                    result["warehouses_full_fulfillment"].append({
                        "warehouse_id": warehouse['warehouse_id'],
                        "warehouse_name": warehouse['warehouse_name'],
                        "warehouse_location": warehouse['warehouse_location']
                    })
                elif has_any_availability:
                    warehouse_details["has_partial"] = True
                    result["warehouses_partial_fulfillment"].append({
                        "warehouse_id": warehouse['warehouse_id'],
                        "warehouse_name": warehouse['warehouse_name'],
                        "warehouse_location": warehouse['warehouse_location']
                    })
                
                result["details"].append(warehouse_details)
            
            # Check if any items cannot be fulfilled from any warehouse
            for item in items:
                product_id = item['product_id']
                requested_quantity = item['quantity']
                
                # Get total available quantity across all warehouses
                total_available_query = """
                    SELECT product_id, SUM(available_quantity) as total_available
                    FROM warehouses.inventory
                    WHERE product_id = %s
                    GROUP BY product_id
                """
                cursor.execute(total_available_query, (product_id,))
                total_available = cursor.fetchone()
                
                total_available_qty = total_available['total_available'] if total_available else 0
                
                if total_available_qty < requested_quantity:
                    result["unavailable_items"].append({
                        "product_id": product_id,
                        "requested": requested_quantity,
                        "total_available_across_warehouses": total_available_qty,
                        "shortage": requested_quantity - total_available_qty
                    })
            
            result["can_fulfill_completely"] = len(result["warehouses_full_fulfillment"]) > 0 and len(result["unavailable_items"]) == 0
            
            return result
            
    finally:
        conn.close()

@tool
def reserve_warehouse_items(reservations: list[dict]) -> dict:
    
    """Reserve items from multiple warehouses in a single transaction.
    
    Args:
        reservations: A list of reservations. Each reservation is a dictionary with keys:
                     - warehouse_id: The warehouse to reserve from
                     - product_id: The product to reserve
                     - quantity: The quantity to reserve
        
    Returns:
        A dictionary containing:
        - success: bool indicating if all reservations were successful
        - reserved_items: list of successfully reserved items
        - failed_items: list of items that could not be reserved
    """
    
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="tools_database",
        user="tools_user",
        password="tools_user_password"
    )
    conn.autocommit = False  # Use transaction
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            result = {
                "success": False,
                "reserved_items": [],
                "failed_items": []
            }
            
            for reservation in reservations:
                warehouse_id = reservation['warehouse_id']
                product_id = reservation['product_id']
                quantity = reservation['quantity']
                
                # Check and lock the inventory row
                check_query = """
                    SELECT warehouse_id, product_id, warehouse_name, warehouse_location, 
                           total_quantity, reserved_quantity, available_quantity
                    FROM warehouses.inventory
                    WHERE warehouse_id = %s AND product_id = %s
                    FOR UPDATE
                """
                cursor.execute(check_query, (warehouse_id, product_id))
                inventory = cursor.fetchone()
                
                if inventory and inventory['available_quantity'] >= quantity:
                    # Update inventory to reserve the items
                    update_query = """
                        UPDATE warehouses.inventory
                        SET reserved_quantity = reserved_quantity + %s
                        WHERE warehouse_id = %s AND product_id = %s
                    """
                    cursor.execute(update_query, (quantity, warehouse_id, product_id))
                    
                    result["reserved_items"].append({
                        "product_id": product_id,
                        "quantity": quantity,
                        "warehouse_id": warehouse_id,
                        "warehouse_name": inventory['warehouse_name'],
                        "warehouse_location": inventory['warehouse_location']
                    })
                else:
                    result["failed_items"].append({
                        "product_id": product_id,
                        "warehouse_id": warehouse_id,
                        "requested": quantity,
                        "available": inventory['available_quantity'] if inventory else 0,
                        "reason": "insufficient_stock" if inventory else "not_in_warehouse"
                    })
            
            # Only commit if all items were successfully reserved
            if len(result["failed_items"]) == 0:
                conn.commit()
                result["success"] = True
            else:
                conn.rollback()
                result["success"] = False
            
            return result
            
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()