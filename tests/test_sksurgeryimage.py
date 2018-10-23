# coding=utf-8

"""scikit-surgeryimage tests"""

from sksurgeryimage.ui.sksurgeryimage_demo import run_demo


def test_using_pytest_sksurgeryimage():
    assert run_demo(True, "Hello World") == True

