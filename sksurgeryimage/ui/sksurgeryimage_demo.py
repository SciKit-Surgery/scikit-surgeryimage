# coding=utf-8

"""Hello world demo module"""
import six


def run_demo(console, text):
    """Shows text message either on console or in a TkInter window."""

    six.print_(text)

    if not console:
        from tkinter import Tk, Label

        root = Tk()

        label = Label(root, text=text)
        label.pack()

        root.mainloop()

    return True
