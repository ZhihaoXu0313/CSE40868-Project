"""
Parameter space definition and utilities for TCAD optimization.
"""

from typing import Dict, Tuple, Union, Any, Optional
import jax.numpy as jnp
import jax
from jax import random
import numpy as np


class ParameterSpace:
    """
    Defines the parameter space for TCAD optimization.
    
    Handles parameter bounds, transformations, and sampling.
    """
    
    def __init__(self, bounds: Dict[str, Tuple[float, float]], 
                 transforms: Optional[Dict[str, str]] = None):
        """
        Initialize parameter space.
        
        Args:
            bounds: Dictionary mapping parameter names to (min, max) bounds
            transforms: Optional transformations ('log', 'linear') for each parameter
        """
        self.bounds = bounds
        self.param_names = list(bounds.keys())
        self.n_params = len(self.param_names)
        
        # Set up transformations (log-scale for parameters spanning orders of magnitude)
        self.transforms = transforms or {}
        for name in self.param_names:
            if name not in self.transforms:
                # Auto-detect if log transform is beneficial
                min_val, max_val = bounds[name]
                if min_val > 0 and max_val / min_val > 100:
                    self.transforms[name] = 'log'
                else:
                    self.transforms[name] = 'linear'
        
        # Precompute transformed bounds for efficiency
        self._compute_transformed_bounds()
    
    def _compute_transformed_bounds(self):
        """Compute bounds in transformed space."""
        self.transformed_bounds = {}
        for name, (min_val, max_val) in self.bounds.items():
            if self.transforms[name] == 'log':
                self.transformed_bounds[name] = (jnp.log(min_val), jnp.log(max_val))
            else:
                self.transformed_bounds[name] = (min_val, max_val)
    
    def sample_uniform(self, key: jax.random.PRNGKey, n_samples: int = 1) -> Dict[str, jnp.ndarray]:
        """
        Sample parameters uniformly from the parameter space.
        
        Args:
            key: JAX random key
            n_samples: Number of samples to generate
            
        Returns:
            Dictionary of sampled parameters
        """
        keys = random.split(key, self.n_params)
        samples = {}
        
        for i, name in enumerate(self.param_names):
            min_val, max_val = self.transformed_bounds[name]
            sample = random.uniform(keys[i], (n_samples,), minval=min_val, maxval=max_val)
            
            # Transform back to original space
            if self.transforms[name] == 'log':
                sample = jnp.exp(sample)
            
            samples[name] = sample
        
        return samples
    
    def dict_to_vector(self, params: Dict[str, Union[float, jnp.ndarray]]) -> jnp.ndarray:
        """
        Convert parameter dictionary to vector form.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Parameter vector in transformed space
        """
        vector = []
        for name in self.param_names:
            val = params[name]
            if self.transforms[name] == 'log':
                val = jnp.log(val)
            vector.append(val)
        return jnp.array(vector)
    
    def vector_to_dict(self, vector: jnp.ndarray) -> Dict[str, float]:
        """
        Convert parameter vector to dictionary form.
        
        Args:
            vector: Parameter vector in transformed space
            
        Returns:
            Parameter dictionary in original space
        """
        params = {}
        for i, name in enumerate(self.param_names):
            val = vector[i]
            if self.transforms[name] == 'log':
                val = jnp.exp(val)
            params[name] = float(val)
        return params
    
    def clip_to_bounds(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Clip parameters to stay within bounds.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Clipped parameter dictionary
        """
        clipped = {}
        for name, value in params.items():
            min_val, max_val = self.bounds[name]
            clipped[name] = float(jnp.clip(value, min_val, max_val))
        return clipped
    
    def get_bounds_vector(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get parameter bounds as vectors.
        
        Returns:
            Tuple of (lower_bounds, upper_bounds) in transformed space
        """
        lower = []
        upper = []
        for name in self.param_names:
            min_val, max_val = self.transformed_bounds[name]
            lower.append(min_val)
            upper.append(max_val)
        return jnp.array(lower), jnp.array(upper)
    
    def scale_to_unit(self, vector: jnp.ndarray) -> jnp.ndarray:
        """
        Scale parameter vector to [0, 1] range.
        
        Args:
            vector: Parameter vector in transformed space
            
        Returns:
            Scaled vector in [0, 1]
        """
        lower, upper = self.get_bounds_vector()
        return (vector - lower) / (upper - lower)
    
    def scale_from_unit(self, scaled_vector: jnp.ndarray) -> jnp.ndarray:
        """
        Scale parameter vector from [0, 1] range back to original bounds.
        
        Args:
            scaled_vector: Parameter vector in [0, 1] range
            
        Returns:
            Vector in transformed parameter space
        """
        lower, upper = self.get_bounds_vector()
        return scaled_vector * (upper - lower) + lower
