# e2e_ai_engineering_bootcamp
This repo is dedicated to including all the practical exercises done during the End-to-Eng AI Engineering Bootcamp (by Aurimas), Cohort 5. Each week will have its own branch.

## Set up
- Clone the repo
- Install uv (https://docs.astral.sh/uv/getting-started/installation/)
- Run `uv sync` to install the dependencies and create the virtual environment under as `.venv` folder
- Create a `.env` file in the root folder with your own API keys and settings based on the `.env.example` file 
- If you want to run the notebooks, go to the `notebooks` folder, choose the notebook you want to run. Then, select the kernel based on the virtual environment you created and feel free to run the cells as needed
- If you want to run the containerized application, run `make run-docker-compose` from the root folder to start the containers, which will start the Streamlit app, the API and the Qdrant vector DB
- If you want to run the evaluations, run `make run-eval-retriever` from the root folder to run the retriever evaluation

## Week 1
- Understand the AI product lifecycle
- Tooling Overview
- What is RAG?
- Embedding models and vector DB
- Implement a RAG pipeline with observability
- RAG pipeline evaluation

## Contact me:
- [LinkedIn](https://www.linkedin.com/in/pedrocrespo94/)