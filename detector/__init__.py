"""
Description for Package
"""
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from .detector import Detector, Factory
from .visualize import visualize, generate_random_color
