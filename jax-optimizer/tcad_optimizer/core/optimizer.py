"""
Main TCAD optimizer class that coordinates different optimization algorithms.
"""

from typing import Dict, Union, Optional, Any, Callable, List, Tuple
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from dataclasses import dataclass
import time
from enum import Enum

from .parameter_space import ParameterSpace
from .cost_function import CostFunctionWrapper


class OptimizationAlgorithm(Enum):
    """Available optimization algorithms."""
    ADAM = "adam"
    RMSPROP = "rmsprop"
    SGD = "sgd"
    LBFGS = "lbfgs"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    SIMULATED_ANNEALING = "simulated_annealing"
    PARTICLE_SWARM = "particle_swarm"
    HYBRID_DE_ADAM = "hybrid_de_adam"


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    params: Dict[str, float]
    cost: float
    n_iterations: int
    n_evaluations: int
    convergence_history: List[float]
    parameter_history: List[Dict[str, float]]
    success: bool
    message: str
    optimization_time: float


class TCADOptimizer:
    """
    Main optimizer class for TCAD applications.
    
    Supports multiple optimization algorithms and provides a unified interface.
    """
    
    def __init__(self, 
                 cost_function: Callable[[Dict[str, float]], float],
                 parameter_space: ParameterSpace,
                 algorithm: Union[str, OptimizationAlgorithm] = OptimizationAlgorithm.ADAM,
                 algorithm_params: Optional[Dict[str, Any]] = None,
                 random_seed: int = 42):
        """
        Initialize TCAD optimizer.
        
        Args:
            cost_function: Function to minimize, takes parameter dict, returns scalar
            parameter_space: Parameter space definition
            algorithm: Optimization algorithm to use
            algorithm_params: Algorithm-specific parameters
            random_seed: Random seed for reproducibility
        """
        self.parameter_space = parameter_space
        self.cost_wrapper = CostFunctionWrapper(cost_function, parameter_space)
        
        # Set algorithm
        if isinstance(algorithm, str):
            self.algorithm = OptimizationAlgorithm(algorithm)
        else:
            self.algorithm = algorithm
        
        # Set default algorithm parameters
        self.algorithm_params = self._get_default_params()
        if algorithm_params:
            self.algorithm_params.update(algorithm_params)
        
        # Initialize random key
        self.key = random.PRNGKey(random_seed)
        
        # Import algorithm modules
        self._import_algorithms()
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for each algorithm."""
        defaults = {
            OptimizationAlgorithm.ADAM: {
                'learning_rate': 0.01,
                'beta1': 0.9,
                'beta2': 0.999,
                'eps': 1e-8
            },
            OptimizationAlgorithm.RMSPROP: {
                'learning_rate': 0.01,
                'decay': 0.9,
                'eps': 1e-8
            },
            OptimizationAlgorithm.SGD: {
                'learning_rate': 0.01,
                'momentum': 0.0
            },
            OptimizationAlgorithm.LBFGS: {
                'learning_rate': 1.0,
                'history_size': 10
            },
            OptimizationAlgorithm.DIFFERENTIAL_EVOLUTION: {
                'population_size': 50,
                'mutation_factor': 0.8,
                'crossover_probability': 0.9
            },
            OptimizationAlgorithm.SIMULATED_ANNEALING: {
                'initial_temperature': 1000.0,
                'cooling_rate': 0.95,
                'min_temperature': 1e-6
            },
            OptimizationAlgorithm.PARTICLE_SWARM: {
                'population_size': 50,
                'inertia_weight': 0.729,
                'cognitive_coeff': 1.49445,
                'social_coeff': 1.49445
            },
            OptimizationAlgorithm.HYBRID_DE_ADAM: {
                'de_population': 30,
                'de_generations': 50,
                'adam_learning_rate': 0.01
            }
        }
        return defaults.get(self.algorithm, {})
    
    def _import_algorithms(self):
        """Import algorithm implementations."""
        from ..algorithms.gradient_based import GradientBasedOptimizer
        from ..algorithms.differential_evolution import DifferentialEvolution
        from ..algorithms.simulated_annealing import SimulatedAnnealing
        from ..algorithms.hybrid import HybridOptimizer
        
        self.gradient_optimizer = GradientBasedOptimizer()
        self.de_optimizer = DifferentialEvolution()
        self.sa_optimizer = SimulatedAnnealing()
        self.hybrid_optimizer = HybridOptimizer()
    
    def optimize(self, 
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 initial_params: Optional[Dict[str, float]] = None,
                 callback: Optional[Callable] = None,
                 verbose: bool = True) -> OptimizationResult:
        """
        Run optimization.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            initial_params: Initial parameter guess (optional)
            callback: Callback function called each iteration
            verbose: Whether to print progress
            
        Returns:
            OptimizationResult object
        """
        start_time = time.time()
        
        # Generate initial parameters if not provided
        if initial_params is None:
            self.key, subkey = random.split(self.key)
            initial_samples = self.parameter_space.sample_uniform(subkey, 1)
            initial_params = {name: float(val[0]) for name, val in initial_samples.items()}
        
        # Ensure initial params are within bounds
        initial_params = self.parameter_space.clip_to_bounds(initial_params)
        
        if verbose:
            print(f"Starting optimization with {self.algorithm.value}")
            print(f"Initial parameters: {initial_params}")
            initial_cost = self.cost_wrapper.evaluate(initial_params)
            print(f"Initial cost: {initial_cost:.6f}")
        
        # Route to appropriate algorithm
        if self.algorithm in [OptimizationAlgorithm.ADAM, OptimizationAlgorithm.RMSPROP, 
                             OptimizationAlgorithm.SGD, OptimizationAlgorithm.LBFGS]:
            result = self._optimize_gradient_based(
                initial_params, max_iterations, tolerance, callback, verbose
            )
        elif self.algorithm == OptimizationAlgorithm.DIFFERENTIAL_EVOLUTION:
            result = self._optimize_differential_evolution(
                max_iterations, tolerance, callback, verbose
            )
        elif self.algorithm == OptimizationAlgorithm.SIMULATED_ANNEALING:
            result = self._optimize_simulated_annealing(
                initial_params, max_iterations, tolerance, callback, verbose
            )
        elif self.algorithm == OptimizationAlgorithm.HYBRID_DE_ADAM:
            result = self._optimize_hybrid(
                max_iterations, tolerance, callback, verbose
            )
        else:
            raise ValueError(f"Algorithm {self.algorithm} not implemented")
        
        result.optimization_time = time.time() - start_time
        
        if verbose:
            print(f"\nOptimization completed in {result.optimization_time:.2f} seconds")
            print(f"Final parameters: {result.params}")
            print(f"Final cost: {result.cost:.6f}")
            print(f"Function evaluations: {result.n_evaluations}")
        
        return result
    
    def _optimize_gradient_based(self, initial_params, max_iterations, tolerance, callback, verbose):
        """Run gradient-based optimization."""
        return self.gradient_optimizer.optimize(
            self.cost_wrapper, self.parameter_space, initial_params,
            self.algorithm.value, self.algorithm_params,
            max_iterations, tolerance, callback, verbose
        )
    
    def _optimize_differential_evolution(self, max_iterations, tolerance, callback, verbose):
        """Run differential evolution optimization."""
        self.key, subkey = random.split(self.key)
        return self.de_optimizer.optimize(
            self.cost_wrapper, self.parameter_space, subkey,
            self.algorithm_params, max_iterations, tolerance, callback, verbose
        )
    
    def _optimize_simulated_annealing(self, initial_params, max_iterations, tolerance, callback, verbose):
        """Run simulated annealing optimization."""
        self.key, subkey = random.split(self.key)
        return self.sa_optimizer.optimize(
            self.cost_wrapper, self.parameter_space, initial_params, subkey,
            self.algorithm_params, max_iterations, tolerance, callback, verbose
        )
    
    def _optimize_hybrid(self, max_iterations, tolerance, callback, verbose):
        """Run hybrid optimization."""
        self.key, subkey = random.split(self.key)
        return self.hybrid_optimizer.optimize(
            self.cost_wrapper, self.parameter_space, subkey,
            self.algorithm_params, max_iterations, tolerance, callback, verbose
        )
    
    def multi_start_optimization(self, 
                                n_starts: int = 5,
                                max_iterations: int = 1000,
                                tolerance: float = 1e-6,
                                verbose: bool = True) -> OptimizationResult:
        """
        Run multiple optimization starts and return the best result.
        
        Args:
            n_starts: Number of random starts
            max_iterations: Maximum iterations per start
            tolerance: Convergence tolerance
            verbose: Whether to print progress
            
        Returns:
            Best optimization result
        """
        best_result = None
        best_cost = float('inf')
        
        if verbose:
            print(f"Running multi-start optimization with {n_starts} starts")
        
        for i in range(n_starts):
            if verbose:
                print(f"\nStart {i+1}/{n_starts}")
            
            # Generate random initial parameters
            self.key, subkey = random.split(self.key)
            initial_samples = self.parameter_space.sample_uniform(subkey, 1)
            initial_params = {name: float(val[0]) for name, val in initial_samples.items()}
            
            # Run optimization
            result = self.optimize(
                max_iterations=max_iterations,
                tolerance=tolerance,
                initial_params=initial_params,
                verbose=False
            )
            
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
                if verbose:
                    print(f"New best cost: {best_cost:.6f}")
        
        if verbose:
            print(f"\nBest result from {n_starts} starts:")
            print(f"Cost: {best_result.cost:.6f}")
            print(f"Parameters: {best_result.params}")
        
        return best_result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = self.cost_wrapper.get_statistics()
        return {
            'algorithm': self.algorithm.value,
            'algorithm_params': self.algorithm_params,
            **stats
        }
