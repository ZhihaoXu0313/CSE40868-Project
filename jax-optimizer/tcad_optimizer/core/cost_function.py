"""
Cost function utilities and wrappers for TCAD optimization.
"""

from typing import Dict, Callable, Any, Optional, Tuple
import jax
import jax.numpy as jnp
from functools import partial
import time


class CostFunctionWrapper:
    """
    Wrapper for cost functions to handle parameter transformations and caching.
    """
    
    def __init__(self, 
                 cost_function: Callable[[Dict[str, float]], float],
                 parameter_space: 'ParameterSpace',
                 enable_caching: bool = True,
                 cache_tolerance: float = 1e-8):
        """
        Initialize cost function wrapper.
        
        Args:
            cost_function: The original cost function taking parameter dict
            parameter_space: Parameter space for transformations
            enable_caching: Whether to enable result caching
            cache_tolerance: Tolerance for cache hit detection
        """
        self.cost_function = cost_function
        self.parameter_space = parameter_space
        self.enable_caching = enable_caching
        self.cache_tolerance = cache_tolerance
        
        # Cache for function evaluations
        self._cache = {} if enable_caching else None
        self.n_evaluations = 0
        self.evaluation_times = []
        
        # Create JAX-compatible versions
        self._create_jax_functions()
    
    def _create_jax_functions(self):
        """Create JAX-compatible function versions."""
        
        @jax.jit
        def vector_cost_function(vector: jnp.ndarray) -> float:
            """Cost function that takes parameter vector."""
            params_dict = self.parameter_space.vector_to_dict(vector)
            return self._evaluate_with_caching(params_dict)
        
        self.vector_cost_function = vector_cost_function
        
        # Create gradient function
        self.grad_fn = jax.jit(jax.grad(vector_cost_function))
        
        # Create hessian function  
        self.hessian_fn = jax.jit(jax.hessian(vector_cost_function))
        
        # Value and gradient function
        self.value_and_grad_fn = jax.jit(jax.value_and_grad(vector_cost_function))
    
    def _evaluate_with_caching(self, params: Dict[str, float]) -> float:
        """Evaluate cost function with optional caching."""
        start_time = time.time()
        
        if self.enable_caching:
            # Create cache key from parameter values
            cache_key = tuple(sorted(params.items()))
            
            # Check cache
            for cached_key, cached_result in self._cache.items():
                if self._params_close(dict(cache_key), dict(cached_key)):
                    return cached_result
        
        # Evaluate function
        result = float(self.cost_function(params))
        
        # Update statistics
        self.n_evaluations += 1
        self.evaluation_times.append(time.time() - start_time)
        
        # Cache result
        if self.enable_caching:
            self._cache[cache_key] = result
            
            # Limit cache size
            if len(self._cache) > 10000:
                # Remove oldest entries
                keys_to_remove = list(self._cache.keys())[:1000]
                for key in keys_to_remove:
                    del self._cache[key]
        
        return result
    
    def _params_close(self, params1: Dict[str, float], params2: Dict[str, float]) -> bool:
        """Check if two parameter sets are close enough for cache hit."""
        for key in params1:
            if key not in params2:
                return False
            if abs(params1[key] - params2[key]) > self.cache_tolerance:
                return False
        return True
    
    def evaluate(self, params: Dict[str, float]) -> float:
        """
        Evaluate cost function for parameter dictionary.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Cost value
        """
        return self._evaluate_with_caching(params)
    
    def evaluate_vector(self, vector: jnp.ndarray) -> float:
        """
        Evaluate cost function for parameter vector.
        
        Args:
            vector: Parameter vector in transformed space
            
        Returns:
            Cost value
        """
        return self.vector_cost_function(vector)
    
    def gradient(self, vector: jnp.ndarray) -> jnp.ndarray:
        """
        Compute gradient with respect to parameter vector.
        
        Args:
            vector: Parameter vector in transformed space
            
        Returns:
            Gradient vector
        """
        return self.grad_fn(vector)
    
    def hessian(self, vector: jnp.ndarray) -> jnp.ndarray:
        """
        Compute Hessian matrix with respect to parameter vector.
        
        Args:
            vector: Parameter vector in transformed space
            
        Returns:
            Hessian matrix
        """
        return self.hessian_fn(vector)
    
    def value_and_grad(self, vector: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
        """
        Compute both value and gradient efficiently.
        
        Args:
            vector: Parameter vector in transformed space
            
        Returns:
            Tuple of (value, gradient)
        """
        return self.value_and_grad_fn(vector)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get evaluation statistics.
        
        Returns:
            Dictionary with evaluation statistics
        """
        return {
            'n_evaluations': self.n_evaluations,
            'total_time': sum(self.evaluation_times),
            'avg_time_per_eval': jnp.mean(jnp.array(self.evaluation_times)) if self.evaluation_times else 0,
            'cache_size': len(self._cache) if self._cache else 0
        }
    
    def reset_statistics(self):
        """Reset evaluation statistics."""
        self.n_evaluations = 0
        self.evaluation_times = []
        if self._cache:
            self._cache.clear()


def create_surrogate_cost_function(surrogate_model, target_specs: Dict[str, float]) -> Callable:
    """
    Create a cost function using a surrogate model.
    
    Args:
        surrogate_model: Trained surrogate model
        target_specs: Target specifications to optimize towards
        
    Returns:
        Cost function that can be used with the optimizer
    """
    def cost_function(params: Dict[str, float]) -> float:
        # Get predictions from surrogate model
        predictions = surrogate_model.predict(params)
        
        # Calculate cost based on difference from targets
        cost = 0.0
        for spec_name, target_value in target_specs.items():
            if spec_name in predictions:
                predicted_value = predictions[spec_name]
                # Normalized squared error
                cost += ((predicted_value - target_value) / target_value) ** 2
        
        return cost
    
    return cost_function
