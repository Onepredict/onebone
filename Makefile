setup:
	git config commit.template .gitmessage.txt
	poetry install
	pre-commit install

format:
	isort .
	black .

test:
	isort . --check
	black . --check

	rm -rf .pytest_cache
	flake8 .
	python -m pytest tests --cov oplib

pre-commit-test:
	sh scripts/pre_commit_test.sh

build:
	pip install build
	python -m build .

sphinx-start:
	mkdir docs
	sphinx quick-start docs

sphinx-build:
	rm -rf docs/build/html
	find docs/source -type f -name "*.rst" -not -name "index.rst" -delete
	sphinx-apidoc -fMT -o docs/source oplib -t docs/templates
	sphinx-build -b html docs/source/ docs/build/html

sphinx-server-docker-build:
	docker build -t sphinx-server -f Dockerfiles/document_server.Dockerfile .

sphinx-server-docker-run:
	docker-compose -f Dockerfiles/docker-compose.yml up -d

sphinx-server-docker-stop:
	docker-compose -f Dockerfiles/docker-compose.yml down