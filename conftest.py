"""
Ensures the project root is on sys.path so `import backend...` / `import
ml...` work regardless of how pytest is invoked, and standardises the
working directory for tests that read relative file paths (e.g.
tests/test_frontend.py reading frontend/templates/*.html).
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)
