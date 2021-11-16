setup:
	git config commit.template .gitmessage.txt

format:
	isort .
	black .

test:
	isort . --check
	black . --check

	flake8 .