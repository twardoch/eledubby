#!/usr/bin/env python3
# this_file: src/eledubby/__main__.py
"""Make eledubby package executable as a module."""

import sys

import fire

from eledubby.eledubby import (
    cast,
    checkpoints,
    dub,
    fx,
    plugin_params,
    plugins,
    presets,
    preview,
    quality,
    recover,
    voices,
)


def cli():
    commands = {
        "dub": dub,
        "fx": fx,
        "cast": cast,
        "voices": voices,
        "plugins": plugins,
        "plugin-params": plugin_params,
        "presets": presets,
        "checkpoints": checkpoints,
        "preview": preview,
        "recover": recover,
        "quality": quality,
    }
    if len(sys.argv) >= 2 and sys.argv[1].startswith("-") and sys.argv[1] not in {"--help", "-h"}:
        fire.Fire(dub)
    else:
        fire.Fire(commands)


if __name__ == "__main__":
    cli()
