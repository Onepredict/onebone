setup:
	git config commit.template .gitmessage.txt
	pip install -r requirements.txt
	pre-commit install

format:
	isort .
	black .

test:
	isort . --check
	black . --check

	flake8 .