import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--update-ref-values", 
        action="store_true", 
        default=False,
        help="Update reference values instead of asserting"
    )


def pytest_collection_modifyitems(config, items):
    """
    If --update-ref-values flag supplied to pytest, skip everything except the
    fixture and test you need to update the reference values.
    """
    update = config.getoption("--update-ref-values")
    if not update:
        return
    
    skip_marker = pytest.mark.skip(
        reason = "Skipped because --update-ref-vales is True"
    )

    for item in items:
        if "test_extraction_quality" not in item.nodeid:
            item.add_marker(skip_marker)
