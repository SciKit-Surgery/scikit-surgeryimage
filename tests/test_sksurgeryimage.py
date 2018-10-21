# coding=utf-8

"""scikit-surgeryimage tests"""

from sksurgeryimage.ui.sksurgeryimage_demo import run_demo

# Unittest style test case
from unittest import TestCase

class Testsksurgeryimage(TestCase):
    def test_using_unittest_sksurgeryimage(self):
        return_value = run_demo(True, "Hello world")
        self.assertTrue(return_value)


# Pytest style

def test_using_pytest_sksurgeryimage():
    assert run_demo(True, "Hello World") == True

