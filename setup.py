import os

from mypyc.build import mypycify
from setuptools import setup

EXT_MODULES = []
if os.getenv("ENABLE_MYPY", False):
    # This currently not work.
    # Just leave code here for future?
    EXT_MODULES.extend(
        mypycify(
            [
                "src/models/lightgcn.py",
            ]
        )
    )


setup(ext_modules=EXT_MODULES)
