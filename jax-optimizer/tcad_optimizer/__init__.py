"""
JAX-based TCAD Optimizer Package

A comprehensive optimization framework for TCAD applications.
"""

from .core.optimizer import TCADOptimizer
from .core.parameter_space import ParameterSpace
from .core.cost_function import CostFunctionWrapper

__version__ = "0.1.0"
__author__ = "TCAD Optimization Team"

__all__ = [
    "TCADOptimizer",
    "ParameterSpace", 
    "CostFunctionWrapper"
]
