Welcome, Onebone!
=================

`Onebone <https://github.com/Onepredict/onebone>`_ is an open-source software for signal analysis about predictive maintenance, 
being used for research activities at `ⓒ ONEPREDICT Corp. <https://onepredict.ai/>`_. 
It includes modules for preprocessing, health feature, and more. 
If you need to analyze signals for industrial equipments like turbines, a rotary machinery or 
componets like gears, bearings, give onebone a try!


Getting Started
===============

**1. Prerequisite**

onebone requires Python 3.6.5+.

**2. Installation**

onebone can be installed via pip from `PyPI <https://pypi.org/project/onebone/>`_.

.. code-block:: bash

   $ pip install onebone


It can be checked as follows whether the onebone has been installed.

.. code-block:: python

   >>> import onebone
   >>> onebone.__version__


**3. Usage**

It assumes that the user has already installed the onebone package.
You can import directly the function, for example:

.. code-block:: python

   >>> from onebone.feature import tacho_to_rpm


Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   onebone


Call for contribute
===================

We appreciate and welcome contributions. Small improvements or fixes are always appreciated; issues labeled as "good first issue" may be a good starting point.

Writing code isn't the only way to contribute to onebone. You can also:

- triage issues
- review pull requests
- help with outreach and onboard new contributors

If you're unsure where to start or how your skills fit in, reach out! You can ask here, on GitHub, by leaving a comment on a relevant issue that is already open.

If you want to use an code for signal analysis, but it's not in onebone, make a issue.

Please follow `this guide <https://github.com/Onepredict/onebone/blob/main/wiki/development_guide.md>`_ to contribute to onebone.