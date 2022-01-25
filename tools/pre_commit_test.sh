isort . --check
black . --check

flake8 .
# python -m pytest tests --cov onebone
poetry run pytest --cov=onebone