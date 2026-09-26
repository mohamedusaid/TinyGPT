"""
Standalone CLI script to audit and print exact TinyGPT-500M parameter allocations.
"""

import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.config import TinyGPTConfig

if __name__ == "__main__":
    config = TinyGPTConfig()
    config.print_audit()
