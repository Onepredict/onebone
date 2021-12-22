isort . --check
black . --check

flake8 .
# python -m pytest tests --cov oplib
poetry run pytest --cov=oplib