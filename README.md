# e2e_ai_engineering_bootcamp
This repo is dedicated to including all the practical exercises done during the End-to-Eng AI Engineering Bootcamp (by Aurimas), Cohort 5. Each week will have its own branch.

## Set up
- Clone the repo
- Install python (at least version 3.11 - https://realpython.com/installing-python/)
- Install docker (https://docs.docker.com/get-docker/)
- Install uv (https://docs.astral.sh/uv/getting-started/installation/)
- Run `uv sync` to install the dependencies and create the virtual environment under as `.venv` folder
- Create a `.env` file in the root folder with your own API keys and settings based on the `.env.example` file 
- If you want to run the notebooks, go to the `notebooks` folder, choose the notebook you want to run. Then, select the kernel based on the virtual environment you created and feel free to run the cells as needed
- If you want to run the containerized application, run `make run-docker-compose` from the root folder to start the containers, which will start the Streamlit app, the API, Qdrant vector DB and Postgres
- If you want to run the evaluations, run `make run-eval-retriever` from the root folder to run the retriever evaluation

## Week 1
- Understand the AI product lifecycle
- Tooling Overview
- What is RAG?
- Embedding models and vector DB
- Implement a RAG pipeline with observability
- RAG pipeline evaluation

## Week 2
- Hybrid Vector Search
- Prompt Management with YAML, Jinja and LangSmith Prompt Registry

## Week 3
- Query Rewriting
- Tool using and ReAct Agent in LangGraph
- Routing Pattern

## Week 4
- State persistence
- Human-in-the-loop
- MCP Server and MCP tools
- State Streaming

## Week 5
- BREAK

## Week 6
- Multi-agents architecture
- Database management as tools

## Week7
- Coordinator Agent Evals
- A2A with LangGraph and MAS

## Contact me:
- [LinkedIn](https://www.linkedin.com/in/pedrocrespo94/)