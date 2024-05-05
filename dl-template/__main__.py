import pkgutil
import os
import importlib

from .env import cli, IN_NOTEBOOK

"""
https://github.com/Nugine/dl-template
"""


if __name__ == "__main__" and not IN_NOTEBOOK:
    for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
        if module_name == "__main__":
            continue
        importlib.import_module(f"{__package__}.{module_name}")

    cli()
