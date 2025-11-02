"""OCEC: Open and closed eye classification pipeline."""

from .model import OCEC
from .pipeline import main

__all__ = ["OCEC", "main"]
