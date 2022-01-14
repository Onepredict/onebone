<p align="center">
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
<img src="https://user-images.githubusercontent.com/79968466/149269606-6b401104-4a5a-4456-b924-558d233131f2.png" align="center" width="50%">
<!--TODO: Change image more beautiful.-->
<!--TODO: Add badges.-->

<!--TODO: Change the name of this library, oplib.-->
# Summary
`oplib` is an open-source software for signal analysis about predictive maintenance, being used for research activities at [ⓒ ONEPREDICT Corp.](https://onepredict.ai/). It includes modules for preprocessing, health feature, and more. If you need to analyze signals for industrial equipments like turbines, a rotary machinery or componets like gears, bearings, give oplib a try!.

The directory is as follows:
``` text
.
├── docs
├── oplib
│ ├── feature
│ ├── math
│ ├── preprocessing
│ ├── signal
│ └── utils
├── tests
├── tools
├── README.md
├── Makefile
└── pyproject.toml
```
# Contributors

# Getting started

## Prerequisite
oplib requires Python 3.6.5+.

## Installation
<!--TODO: Register oplib on the pypi server.-->
oplib can be installed via pip from [PyPI](https://pypi.org/)
``` bash
$ pip install --extra-index-url http://10.10.30.16:8008 --trusted-host 10.10.30.16:8008 oplib
```
It can be checked as follows whether the oplib has been installed.
``` python
>>> import oplib
>>> oplib.__version__
```

## Usage
It assumes that the user has already installed the oplib package.

You can import directly the function, for example:
``` python
>>> from oplib.feature import tacho_to_rpm
```

# Call for contribute
We appreciate and welcome contributions. Small improvements or fixes are always appreciated; issues labeled as "good first issue" may be a good starting point.

Writing code isn't the only wat to contribute to oplib. You can also:

- triage issues
- review pull requests
- help with outreach and onboard new contributors

If you're unsure where to start or how your skills fit in, reach out! You can ask here, on GitHub, by leaving a comment on a relevant issue that is already open.

If you want to use an code for signal analysis, but it's not in oplib, make a issue.

If you are new to contributing to open source, [this guide](https://opensource.guide/how-to-contribute/) helps explain why, what, and how to get involved.

If you want to contribute oplib, follow [this guide](https://github.com/Onepredict/oplib/blob/main/wiki/development_guide.md).

# References
- [Numpy](https://numpy.org/)
- [Scipy](https://scipy.org/)
- [Matlab](https://www.mathworks.com/help/index.html?s_tid=CRUX_lftnav)

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/Kangwhi-Kim"><img src="https://avatars.githubusercontent.com/u/79968466?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Kangwhi Kim</b></sub></a><br /><a href="https://github.com/Onepredict/oplib/commits?author=Kangwhi-Kim" title="Code">💻</a> <a href="https://github.com/Onepredict/oplib/commits?author=Kangwhi-Kim" title="Documentation">📖</a> <a href="https://github.com/Onepredict/oplib/pulls?q=is%3Apr+reviewed-by%3AKangwhi-Kim" title="Reviewed Pull Requests">👀</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!