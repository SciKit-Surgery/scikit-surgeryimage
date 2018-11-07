scikit-surgeryimage
== == == == == == == == == == == == == == == =

.. image:: https: // weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/scikit-surgeryimage/raw/master/project-icon.png
: height: 128px
: width: 128px
: target: https: // weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/scikit-surgeryimage

.. image:: https: // weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/scikit-surgeryimage/badges/master/build.svg
: target: https: // weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/scikit-surgeryimage/pipelines
: alt: GitLab-CI test status

.. image:: https: // weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/scikit-surgeryimage/badges/master/coverage.svg
: target: https: // weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/scikit-surgeryimage/commits/master
: alt: Test coverage

.. image:: https: // travis-ci.org/WEISS/SoftwareRepositories/scikit-surgeryimage.svg?branch = master
: target: https: // travis-ci.org/WEISS/SoftwareRepositories/scikit-surgeryimage
: alt: Travis test status

.. image:: https: // weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/scikit-surgeryimage/badges/master/coverage.svg
: target: https: // weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/scikit-surgeryimage/pipelines

.. image: : https: // readthedocs.org/projects/scikit-surgeryimage/badge /?version = latest
: target: http: // scikit-surgeryimage.readthedocs.io/en/latest /?badge = latest
: alt: Documentation Status


scikit-surgeryimage is a python project that does interesting things.

Author: Matt Clarkson

scikit-surgeryimage was developed at the `Wellcome EPSRC Centre for Interventional and Surgical Sciences`_ in `University College London(UCL)`_.


Installing
~~~~~~~~~~

You can pip install directly from the repository as follows:
::

    pip install git+https: // weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/scikit-surgeryimage


Developing
^ ^ ^ ^ ^ ^ ^ ^ ^^

You can clone the repository using the following command:

::

    git clone https: // weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/scikit-surgeryimage


Running the tests
^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

You can run the unit tests by installing and running tox:

    ::

        pip install tox
        tox

Get started with PyCharm
^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^^

This assumes you have PyCharm installed and configured to support virtual environments.

1. Start PyCharm
2. Select File > Open
3. Select the project's folder
4. Open in a new window
5. Open Preferences
6. Click on Project: [YourProject] and select Project Interpreter
7. At the right of the Project Interpreterm, click the cog
8. Select Add Local...
9. Select Virtual Environment
10. Choose a location for your virtual environment(for example, [YourHomeFolder]/VirtualEnvs/[YourProjectName])
11. Select a base interpreter(usually the latest version of Python 3).
12. Recommended settings: Do not inherit global site-packages, and do not make available to all projects.
13. Click OK
14. Click on Terminal
15. `pip install tox`
16. `tox`
17. Expand the project
18. Right-click on the Tests folder and choose "Run Unittests in tests". This will create a new configuration for running tests
19. Right-click on sksurgeryimage and select Run sksurgeryimage. This will create a new configuration for running the project.
20. Switch between the program and test configurations using the drop-down at the top of the screen, and the green arrow to run or the green bug to debug.

Contributing
^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^^

Please see the `contributing guidelines`_.


Useful links
^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^^

`Source code repository`_
`Documentation`_


Licensing and copyright
-----------------------

Copyright 2018 University College London.
scikit-surgeryimage is released under the BSD-3 license. Please see the `license file`_ for details.


Acknowledgements
----------------

Supported by `Wellcome`_ and `EPSRC`_.


.. _`Wellcome EPSRC Centre for Interventional and Surgical Sciences`: http: // www.ucl.ac.uk/weiss
.. _`source code repository`: https: // weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/scikit-surgeryimage
.. _`Documentation`: https: // scikit-surgeryimage.readthedocs.io
.. _`University College London(UCL)`: http: // www.ucl.ac.uk/
.. _`Wellcome`: https: // wellcome.ac.uk/
.. _`EPSRC`: https: // www.epsrc.ac.uk/
.. _`contributing guidelines`: https: // weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/scikit-surgeryimage/blob/master/CONTRIBUTING.rst
.. _`license file`: https: // weisslab.cs.ucl.ac.uk/WEISS/SoftwareRepositories/scikit-surgeryimage/blob/master/LICENSE


.. toctree::
    : maxdepth: 4
    : caption: Contents:
