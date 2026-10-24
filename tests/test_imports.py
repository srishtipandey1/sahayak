"""Ensure every Sahayak module imports with the CI dependency extras."""

from importlib import import_module
from pkgutil import walk_packages

import sahayak


def test_every_sahayak_module_imports() -> None:
    module_names = sorted(
        module.name
        for module in walk_packages(sahayak.__path__, prefix=f"{sahayak.__name__}.")
    )

    assert module_names
    for module_name in module_names:
        import_module(module_name)
