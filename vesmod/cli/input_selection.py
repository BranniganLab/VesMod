"""Argparse adapter for one or more input path selectors."""

import argparse


class InputPathsAction(argparse.Action):
    """Preserve one-input Path behavior while accepting multiple selectors."""

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        """Store one Path directly, or multiple Paths as a list."""
        del parser, option_string
        setattr(namespace, self.dest, values[0] if len(values) == 1 else values)
