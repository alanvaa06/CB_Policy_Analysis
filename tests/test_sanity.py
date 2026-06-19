# tests/test_sanity.py
import cbp


def test_package_importable():
    assert hasattr(cbp, "__version__")
