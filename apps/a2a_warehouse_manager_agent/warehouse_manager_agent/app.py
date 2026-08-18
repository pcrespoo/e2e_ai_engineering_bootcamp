import logging
import uvicorn
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    AgentInterface
)
from agent import WarehouseManagerAgent
from dotenv import load_dotenv
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from agent_executor import WarehouseManagerAgentExecutor

from starlette.applications import Starlette
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.routes.agent_card_routes import create_agent_card_routes

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOST = 'localhost'
PORT = 10001

def main():
    capabilities = AgentCapabilities(
        streaming=True,
    )

    skill_availability = AgentSkill(
        id='ABC',
        name='Check Availability',
        description='Check availability of items across warehouses.',
        tags=['warehouse', 'availability'],
        examples=[
            "What is the availability of the item 123 ?",
        ]
    )

    skill_reservation = AgentSkill(
        id='DEF',
        name='Reserve Items',
        description='Reserve items in the warehouse.',
        tags=['warehouse', 'reserve'],
        examples=[
            'Reserve 10 items of the item 123 in Berlin warehouse.',
        ]
    )

    agent_card = AgentCard(
        name='warehouse_manager_agent',
        description='The warehouse manager agent is responsible for checking the availability of items in the warehouse and reserving them.',
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        skills=[skill_availability, skill_reservation],
        capabilities=capabilities,
        supported_interfaces=[
            AgentInterface(
                protocol_binding='JSONRPC',
                protocol_version='1.0',
                url=f'http://{HOST}:{PORT}/'
            )
        ]
    )

    adk_agent = WarehouseManagerAgent().get_agent()
    runner = Runner(
        agent=adk_agent,
        app_name=agent_card.name,
        session_service=InMemorySessionService(),
        artifact_service=InMemoryArtifactService(),
        memory_service=InMemoryMemoryService(),

    )
    agent_executor = WarehouseManagerAgentExecutor(runner)
    request_handler = DefaultRequestHandler(
        agent_card=agent_card,
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore()
    )
    app = Starlette(
        routes=[
            *create_jsonrpc_routes(request_handler, rpc_url='/'), 
            *create_agent_card_routes(agent_card)
        ]
    )
    uvicorn.run(app, host=HOST, port=PORT)

if __name__ == '__main__':
    main()