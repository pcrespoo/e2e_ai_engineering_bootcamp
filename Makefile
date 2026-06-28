run-docker-compose:
	uv sync
	docker-compose up --build

clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/prerequisites/*.ipynb

run-eval-retriever:
	uv sync
	PYTHONPATH=${PWD}/apps/api:${PWD}/apps/api/src:$$PYTHONPATH:${PWD} uv run --env-file .env python -m evals.eval_retriever