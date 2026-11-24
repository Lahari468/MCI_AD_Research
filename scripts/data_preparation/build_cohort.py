#!/usr/bin/env python3
"""
Cohort Construction Script
CLI driver for building verified ADNI MCI 24-month progression cohorts.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cohort.build_cohort import main

if __name__ == "__main__":
    main()
