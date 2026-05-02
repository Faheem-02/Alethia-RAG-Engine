"""Pytest configuration for backend tests.

Ensures repository root is on sys.path so imports like
`from backend.app...` work whether pytest is launched from repository root
or backend directory.
"""

import sys
from pathlib import Path

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))
