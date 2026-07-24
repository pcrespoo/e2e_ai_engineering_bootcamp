from fastmcp import FastMCP
from qdrant_client import QdrantClient
from items_mcp_server.utils import retrieve_items_data, rerank_data, process_context
mcp = FastMCP("items_mcp_server")

@mcp.tool
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

if __name__ == "__main__":
    mcp.run(transport='http', host="0.0.0.0", port=8001)