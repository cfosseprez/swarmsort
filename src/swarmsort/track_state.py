"""
SwarmSort Track State Implementation

This module contains the track handling mecanics
"""

# ============================================================================
# STANDARD IMPORTS
# ============================================================================
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Type, TypeVar, Union, Literal
from dataclasses import dataclass, field
from collections import deque
import numba as nb
from scipy.optimize import linear_sum_assignment
import os
from pathlib import Path
import time
import sys
from dataclasses import dataclass, asdict, field, fields
import gc

# ============================================================================
# LOGGER
# ============================================================================
from loguru import logger

# ============================================================================
# Internal imports
# ============================================================================
from .data_classes import Detection, TrackedObject
from .config import SwarmSortConfig
from .embedding_scaler import EmbeddingDistanceScaler


