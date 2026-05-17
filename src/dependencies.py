"""Shared application dependencies.

This module holds the true module-level singleton for TransactionCategorizer
so that both predict.py and train.py operate on the same in-memory instance.
Importing `categorizer` from here always returns the same object.
"""
from src.ml.model import TransactionCategorizer

# Module-level singleton — instantiated once at import time and shared by all
# route modules that import from this module.
categorizer = TransactionCategorizer()
