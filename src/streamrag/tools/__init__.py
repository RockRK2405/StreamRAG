"""Build-time tools. This is the ONLY package allowed to touch the network (model download).

The corpus and retrieval packages must never import from here at query time; see
tests/test_isolation.py.
"""
