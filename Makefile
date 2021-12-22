setup:
	git config commit.template .gitmessage.txt
	poetry install
	pre-commit install

clean:
	rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info */.pytest_cache .pytest_cache

format:
	isort .
	black .

test:
	isort . --check
	black . --check

	rm -rf .pytest_cache
	flake8 .
	poetry run pytest --cov=oplib

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

docker-compose-run:
	docker-compose -f Dockerfiles/docker-compose.yml up -d

docker-compose-stop:
	docker-compose -f Dockerfiles/docker-compose.yml down