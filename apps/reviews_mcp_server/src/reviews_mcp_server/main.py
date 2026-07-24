from fastmcp import FastMCP
from qdrant_client import QdrantClient
from reviews_mcp_server.utils import retrieve_prefiltered_reviews_data, process_review_context
mcp = FastMCP("items_mcp_server")

@mcp.tool
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
if __name__ == "__main__":
    mcp.run(transport='http', host="0.0.0.0", port=8002)