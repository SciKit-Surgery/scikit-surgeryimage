scikit-surgeryimage
===============================

.. image:: https://github.com/UCL/scikit-surgeryimage/raw/master/project-icon.png
   :height: 128px
   :width: 128px
   :target: https://github.com/UCL/scikit-surgeryimage
   :alt: Logo

.. image:: https://travis-ci.com/UCL/scikit-surgeryimage.svg?branch=master
   :target: https://travis-ci.com/github/UCL/scikit-surgeryimage
   :alt: Travisx CI test status

.. image:: https://coveralls.io/repos/github/UCL/scikit-surgeryimage/badge.svg?branch=master&service=github
    :target: https://coveralls.io/github/UCL/scikit-surgeryimage?branch=master
    :alt: Test coverage

.. image:: https://readthedocs.org/projects/scikit-surgeryimage/badge/?version=latest
    :target: http://scikit-surgeryimage.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status



scikit-surgeryimage is a python only project to implement image processing algorithms
that are useful for image-guided surgery.

scikit-surgeryimage was developed at the `Wellcome EPSRC Centre for Interventional and Surgical Sciences`_ in `University College London (UCL)`_.

Features
--------

requirements.txt specifies a dependency on opencv-contrib-python, so you can use any OpenCV python therein.
In addition, this project provides:

* Video read and write.
* Utilities to detect the number of cameras and prepare text for overlay on video.
* Functions to make charuco boards, and detect charuco markers.
* Interlacing and deinterlacing.
* Convenience wrappers for erosion and dilation.


Installing
~~~~~~~~~~

You can pip install directly from the repository as follows:
::

    pip install scikit-surgeryimage


Developing
^^^^^^^^^^

You can clone the repository using the following command:

::

    git clone https://github.com/UCL/scikit-surgeryimage


Running the tests
^^^^^^^^^^^^^^^^^

You can run the unit tests by installing and running tox:

    ::

      pip install tox
      tox

Encountering Problems?
^^^^^^^^^^^^^^^^^^^^^^
Please check list of `common issues`_.

Contributing
^^^^^^^^^^^^

Please see the `contributing guidelines`_.


Useful links
^^^^^^^^^^^^

* `Source code repository`_
* `Documentation`_


Licensing and copyright
-----------------------

Copyright 2018 University College London.
scikit-surgeryimage is released under the BSD-3 license. Please see the `license file`_ for details.


Acknowledgements
----------------

Supported by `Wellcome`_ and `EPSRC`_.


.. _`Wellcome EPSRC Centre for Interventional and Surgical Sciences`: http://www.ucl.ac.uk/weiss
.. _`source code repository`: https://github.com/UCL/scikit-surgeryimage
.. _`Documentation`: https://scikit-surgeryimage.readthedocs.io
.. _`University College London (UCL)`: http://www.ucl.ac.uk/
.. _`Wellcome`: https://wellcome.ac.uk/
.. _`EPSRC`: https://www.epsrc.ac.uk/
.. _`contributing guidelines`: https://github.com/UCL/scikit-surgeryimage/blob/master/CONTRIBUTING.rst
.. _`license file`: https://github.com/UCL/scikit-surgeryimage/blob/master/LICENSE
.. _`common issues`: https://github.com/UCL/scikit-surgery/wikis/Common-Issues
