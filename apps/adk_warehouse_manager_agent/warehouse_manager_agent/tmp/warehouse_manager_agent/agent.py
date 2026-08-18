### Import Dependencies
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from warehouse_manager_agent.tools import check_warehouse_availability, reserve_warehouse_items
import os

model = LiteLlm(
    model="openai/gpt-5.4-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)
root_agent = Agent(
    name="warehouse_manager_agent",
    model=model,
    tools=[
        check_warehouse_availability,
        reserve_warehouse_items
    ],
    description="""
    The warehouse manager agent is responsible for checking the availability of items in the warehouse and reserving them.
    """,
    instruction="""
    You are a part of the shopping assistant that can manage available inventory in the warehouses.

    ## Instructions

    - As the final answer you should return an answer to the users query in a form of actions performed.
    - You must always check the availability of the items in the warehouses before reserving them.
    - Only reserve items in warehouses if entire order can be reserved or the user has confirmed that they want a partial reservation.
    - If you cannot reserve any items, return an answer that the order cannot be reserved.
    - If you can reserve some items, return an answer that the order can be partially reserved and include the details.
    - If only partial quantity can be reserved in some warehouses, try to combine the required quantity from different warehouses.
    - Try to reserve items from the closest warehouse to the user first if users location is provided.
    - As the final answer you should return an answer in a form of actions performed.
    """
)