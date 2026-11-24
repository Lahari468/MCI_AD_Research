#!/usr/bin/env python3
"""
Cross-Validation & Holdout Partition Generation Script
CLI driver for generating stratified 3-repeat 5-fold splits with a 20% locked internal holdout.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import main

if __name__ == "__main__":
    main()
