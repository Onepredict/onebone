setup:
	git config commit.template .gitmessage.txt
	poetry install
	pre-commit install

clean:
	rm -vrf ./build ./dist ./*.tgz ./*.egg-info */.pytest_cache .pytest_cache
	python3 -Bc "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.py[co]')]"
	python3 -Bc "import pathlib; [p.rmdir() for p in pathlib.Path('.').rglob('__pycache__')]"

format:
	isort .
	black .

test:
	isort . --check
	black . --check

	rm -rf .pytest_cache
	flake8 onebone tests
	poetry run pytest --cov=onebone

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
	sphinx-apidoc -fMT -o docs/source onebone -t docs/templates
	sphinx-build -b html docs/source/ docs/build/html

sphinx-server-docker-build:
	docker build -t sphinx-server -f Dockerfiles/document_server.Dockerfile .

docker-compose-run:
	docker-compose -f Dockerfiles/docker-compose.yml up -d

docker-compose-stop:
	docker-compose -f Dockerfiles/docker-compose.yml down