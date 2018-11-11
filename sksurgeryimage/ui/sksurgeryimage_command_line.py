# coding=utf-8

"""Command line processing"""


import argparse
from sksurgeryimage import __version__
from sksurgeryimage.ui.sksurgeryimage_demo import run_demo


def main(args=None):
    """Entry point for demo scikit-surgeryimage application."""

    parser = argparse.ArgumentParser(description='scikit-surgeryimage')

    parser.add_argument("-t", "--text",
                        required=False,
                        default="This is scikit-surgeryimage",
                        type=str,
                        help="Text to display")

    parser.add_argument("--console", required=False,
                        action='store_true',
                        help="If set, scikit-surgeryimage "
                             "will not bring up a graphical user interface")

    version_string = __version__
    friendly_version_string = version_string if version_string else 'unknown'
    parser.add_argument(
        "-v", "--version",
        action='version',
        version='scikit-surgeryimage version ' + friendly_version_string)

    args = parser.parse_args(args)

    run_demo(args.console, args.text)
