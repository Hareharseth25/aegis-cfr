"""AEGIS: an adaptive game-theoretic poker engine.

The heavy lifting lives in the C++ extension ``aegis_core``; this package adds
Python-side conveniences (OpenSpiel bridge, spot solving API, experiment
utilities).
"""
from . import aegis_core as core  # noqa: F401
from .aegis_core import *  # noqa: F401,F403

__version__ = "1.0.0"
