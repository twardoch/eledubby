#!/usr/bin/env python3
# this_file: src/eledubby/__main__.py
"""Make eledubby package executable as a module."""

import fire

from eledubby.eledubby import dub, fx


def cli():
    fire.Fire({"dub": dub, "fx": fx})


if __name__ == "__main__":
    cli()
