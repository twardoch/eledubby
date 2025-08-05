# this_file: eledubby/src/eledubby/__init__.py
"""Eledubby - Voice dubbing tool using ElevenLabs API."""

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "unknown"
    __version_tuple__ = (0, 0, 0)

__all__ = ["__version__", "__version_tuple__"]
