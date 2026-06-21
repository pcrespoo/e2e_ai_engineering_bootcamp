run-docker-compose:
	uv sync
	docker-compose up --build

clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/prerequisites/*.ipynb