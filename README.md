# Signal analysis algorithm in Onepredict
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
이 레포의 설명을 적어주세요.

# Prerequisite
## Poetry
의존성 및 패키지 관리툴인 [Poetry](https://github.com/python-poetry/poetry)를 사용합니다.
Poetry 설치 (윈도우는 [링크](https://github.com/python-poetry/poetry#installation) 참고)
```
curl -sSL https://install.python-poetry.org | python -
```

## Pyenv
파이썬 버전 관리와 가상환경으로 [Pyenv](https://github.com/pyenv/pyenv)와 `virtualenv`를 사용합니다. `conda`, `venv` 등을 사용해도 됩니다.
```
pyenv install 3.9.0
pyenv virtualenv 3.9.0 oplib
pyenv local oplib
```

## Setup
레포를 사용하기 위한 기본적인 세팅을 합니다. `poetry`로 필요한 패키지들을 설치하고, git commit template을 적용하는 등의 작업을 수행합니다. 아래 명령어를 사용합니다:
```
make setup

# or
sh tools/setup.sh
```
# Auto-format
formatter로 [black](https://github.com/psf/black), [isort](https://github.com/PyCQA/isort)를 사용합니다. 아래 명령어를 통해 사용 가능합니다:
```
sh tools/format.sh

# or
make format
```
## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/kyunghwan-onepredict"><img src="https://avatars.githubusercontent.com/u/86699249?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Kyunghwan Kim</b></sub></a><br /><a href="https://github.com/Onepredict/oplib/commits?author=kyunghwan-onepredict" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!