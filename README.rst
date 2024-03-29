scikit-surgeryimage
====================

.. image:: https://github.com/SciKit-Surgery/scikit-surgeryimage/raw/master/weiss_logo.png
   :height: 128px
   :width: 128px
   :target: https://github.com/SciKit-Surgery/scikit-surgeryimage
   :alt: Logo

|

.. image:: https://github.com/SciKit-Surgery/scikit-surgeryimage/workflows/.github/workflows/ci.yml/badge.svg
   :target: https://github.com/SciKit-Surgery/scikit-surgeryimage/actions
   :alt: GitHub Actions CI status

.. image:: https://coveralls.io/repos/github/SciKit-Surgery/scikit-surgeryimage/badge.svg?branch=master&service=github
    :target: https://coveralls.io/github/SciKit-Surgery/scikit-surgeryimage?branch=master
    :alt: Test coverage

.. image:: https://readthedocs.org/projects/scikit-surgeryimage/badge/?version=latest
    :target: http://scikit-surgeryimage.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/badge/Cite-SciKit--Surgery-informational
   :target: https://doi.org/10.1007/s11548-020-02180-5
   :alt: The SciKit-Surgery paper

.. image:: https://img.shields.io/twitter/url?style=social&url=http%3A%2F%2Fscikit-surgery.org
   :target: https://twitter.com/intent/tweet?screen_name=scikit_surgery&ref_src=twsrc%5Etfw
   :alt: Get in touch via twitter

.. image:: https://img.shields.io/twitter/follow/scikit_surgery?style=social
   :target: https://twitter.com/scikit_surgery?ref_src=twsrc%5Etfw
   :alt: Follow scikit_surgery on twitter




scikit-surgeryimage is a python only project to implement image processing algorithms
that are useful for image-guided surgery.

scikit-surgeryimage was developed at the `Wellcome EPSRC Centre for Interventional and Surgical Sciences`_ in `University College London (UCL)`_.

.. features-start

Features
--------

requirements.txt specifies a dependency on opencv-contrib-python-headless, so you can use any OpenCV python therein.
In addition, this project provides:

* `Convenience classes <https://scikit-surgeryimage.readthedocs.io/en/latest/module_ref.html#data-acquisition>`_ for video read and write.
* `Utilities <https://scikit-surgeryimage.readthedocs.io/en/latest/module_ref.html#calibration-tools>`_ to detect the number of cameras and prepare text for overlay on video.
* A `PointDetector <https://scikit-surgeryimage.readthedocs.io/en/latest/module_ref.html#utilities>`_ interface for video camera calibration and implementations such as OpenCV chessboard, ArUco, ChArUco and combinations.
* Generate `ChArUco <https://scikit-surgeryimage.readthedocs.io/en/latest/module_ref.html#sksurgeryimage.calibration.charuco.make_charuco_board>`_ patterns
* `Interlacing and deinterlacing <https://scikit-surgeryimage.readthedocs.io/en/latest/module_ref.html#video-interlacing-functions>`_.
* Convenience wrappers for `erosion and dilation <https://scikit-surgeryimage.readthedocs.io/en/latest/module_ref.html#module-sksurgeryimage.processing.morphological_operators>`_.

.. features-end

Installing
~~~~~~~~~~

You can pip install directly from the repository as follows:
::

    pip install scikit-surgeryimage


Developing
^^^^^^^^^^

You can clone the repository using the following command:

::

    git clone https://github.com/SciKit-Surgery/scikit-surgeryimage


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
.. _`source code repository`: https://github.com/SciKit-Surgery/scikit-surgeryimage
.. _`Documentation`: https://scikit-surgeryimage.readthedocs.io
.. _`University College London (UCL)`: http://www.ucl.ac.uk/
.. _`Wellcome`: https://wellcome.ac.uk/
.. _`EPSRC`: https://www.epsrc.ac.uk/
.. _`contributing guidelines`: https://github.com/SciKit-Surgery/scikit-surgeryimage/blob/master/CONTRIBUTING.rst
.. _`license file`: https://github.com/SciKit-Surgery/scikit-surgeryimage/blob/master/LICENSE
.. _`common issues`: https://github.com/SciKit-Surgery/scikit-surgery/wikis/Common-Issues
