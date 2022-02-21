# Development Guide

This guide is designed to help you quickly find the information you need about onebone development. If you're new to this and want to start coding ASAP, you've found the right place.

## Development environment

Follow the steps below.

### 1. Poetry (package & dependency manager)

Poetry is a tool for dependency management and packaging in Python. It allows you to declare the libraries your project depends on and it will manage (install/update) them for you. you need to use Poetry for your developement. It is recommended to visit [Poetry](https://python-poetry.org/) website for installation, but we can help you to install it(v.1.1) quickly.

For osx / linux / bashonwindows:

```bash
$ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

For windows powershell:

```powershell
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
```

### 2. Clone git repository

Fork a copy of the main SciPy repository in Github onto your own account and then create your local repository via:

```bash
$ git clone https://github.com/Onepredict/onebone.git onebone
$ cd onebone
```

#### 2-1. Create new branch

You can create new branch to contribute new feature. The branch naming rule is below:

  - `feature/<contents>`: When you add simple features.
  - `hotfix/<contents>`: When you fix critical issues.

The branch name is checked by github action when you create Pull Request.

### 3. Python version and virtualenv manangement

You can use [pyenv](https://github.com/pyenv/pyenv) with [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) plugin, [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), and so on.
If you want to install and use it, visit the website.

The quick usage for pyenv is:

```bash
$ pyenv install {python_version}
$ pyenv virtualenv {python_version} onebone
$ pyenv local onebone
```

### 4. Make setup

If you successfully clone git repository, you can find the a [Makefile](https://opensource.com/article/18/8/what-how-makefile) containing the following contents:

```text
setup:
	git config commit.template .gitmessage.txt
	poetry install
...
```

And, enter the following instructions in the shell:

```
$ make setup
```

_Note_: If you develop in Windows OS, the `make` command may not work. Then, this [link](http://gnuwin32.sourceforge.net/packages/make.htm) will be helpful.

## Documentation

This section is about something you need to know about writing docstrings, which are rendered to produce HTML documentation using [Sphinx](https://www.sphinx-doc.org/en/master/).
Please use the [formatting standard](https://numpydoc.readthedocs.io/en/latest/format.html#format) of `numpydoc` as shown in their [example](https://numpydoc.readthedocs.io/en/latest/example.html#example).

## Unit tests

`onebone` follows the `numpy`'s [Testing Guidelines](https://numpy.org/devdocs/reference/testing.html) which is the definitive guide to writing unit tests of SciPy code.

## You should do something before `git push`

### Auto-Format

The quick usage for auto-format tools are:

```bash
$ black .
$ isort .
```

Or, you can enter the following instructions in the shell:

```bash
$ make format
```

### Test

You should make test code for the function you have been developed. If you want to know testing codes in python, follow the [link](https://realpython.com/python-testing/#testing-your-code).
Assuming that you have added a feature to the A.py file in the directory `{root}/onebone/feature/`, you should create a test code `test_A.py` in the directory `{root}/tests/feature/`.

We use the Python code formatter [black](https://github.com/psf/black), [isort](https://github.com/PyCQA/isort).
And, you can use the make command like this:

```bash
$ make test
```
