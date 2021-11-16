setup:
	git config commit.template .gitmessage.txt
	poetry install

format:
	isort .
	black .

test:
	isort . --check
	black . --check

	rm -rf .pytest_cache
	flake8 .
	python -m pytest tests --cov app

pre-commit-test:
	sh scripts/pre_commit_test.sh