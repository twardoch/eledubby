#!/usr/bin/env python3
# this_file: eledubby/src/eledubby/__main__.py
"""Make eledubby package executable as a module."""

import fire

from eledubby.eledubby import main


def cli():
    fire.Fire(main)


if __name__ == "__main__":
    cli()
