"""
Simulated Annealing optimization algorithm.
"""

from typing import Dict, Any, Optional, Callable, List
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from ..core.optimizer import OptimizationResult


class SimulatedAnnealing:
    """
    Simulated Annealing optimization algorithm.
    
    A probabilistic technique for approximating the global optimum of a function.
    """
    
    def optimize(self,
                 cost_wrapper,
                 parameter_space,
                 initial_params: Dict[str, float],
                 key: jax.random.PRNGKey,
                 algorithm_params: Dict[str, Any],
                 max_iterations: int,
                 tolerance: float,
                 callback: Optional[Callable] = None,
                 verbose: bool = True) -> OptimizationResult:
        """
        Run simulated annealing optimization.
        
        Args:
            cost_wrapper: Cost function wrapper
            parameter_space: Parameter space
            initial_params: Initial parameters
            key: JAX random key
            algorithm_params: SA parameters
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            callback: Optional callback function
            verbose: Whether to print progress
            
        Returns:
            OptimizationResult
        """
        # SA parameters
        initial_temperature = algorithm_params['initial_temperature']
        cooling_rate = algorithm_params['cooling_rate']
        min_temperature = algorithm_params['min_temperature']
        
        # Initialize
        current_vector = parameter_space.dict_to_vector(initial_params)
        current_cost = cost_wrapper.evaluate_vector(current_vector)
        
        best_vector = current_vector.copy()
        best_cost = current_cost
        
        convergence_history = [float(current_cost)]
        parameter_history = [initial_params.copy()]
        
        temperature = initial_temperature
        n_accepted = 0
        n_rejected = 0
        
        if verbose:
            print(f"Initial cost: {current_cost:.6f}")
            print(f"Initial temperature: {temperature:.2f}")
        
        # Annealing loop
        for iteration in range(max_iterations):
            # Check if temperature is too low
            if temperature < min_temperature:
                if verbose:
                    print(f"Temperature dropped below minimum ({min_temperature})")
                break
            
            # Generate neighbor solution
            key, subkey = random.split(key)
            neighbor_vector = self._generate_neighbor(
                subkey, current_vector, temperature, parameter_space
            )
            
            # Evaluate neighbor
            neighbor_cost = cost_wrapper.evaluate_vector(neighbor_vector)
            
            # Acceptance criterion
            delta_cost = neighbor_cost - current_cost
            
            if delta_cost < 0:
                # Accept improvement
                current_vector = neighbor_vector
                current_cost = neighbor_cost
                n_accepted += 1
                
                # Update best solution
                if neighbor_cost < best_cost:
                    best_vector = neighbor_vector.copy()
                    best_cost = neighbor_cost
            else:
                # Accept with probability exp(-delta/T)
                key, subkey = random.split(key)
                acceptance_prob = jnp.exp(-delta_cost / temperature)
                
                if random.uniform(subkey) < acceptance_prob:
                    current_vector = neighbor_vector
                    current_cost = neighbor_cost
                    n_accepted += 1
                else:
                    n_rejected += 1
            
            # Update temperature
            temperature *= cooling_rate
            
            # Store history
            convergence_history.append(float(best_cost))
            best_params = parameter_space.vector_to_dict(best_vector)
            parameter_history.append(best_params.copy())
            
            # Check convergence
            if len(convergence_history) > 100:
                recent_costs = convergence_history[-100:]
                cost_std = jnp.std(jnp.array(recent_costs))
                if cost_std < tolerance:
                    success = True
                    message = f"Converged after {iteration+1} iterations (std < {tolerance})"
                    break
            
            # Callback
            if callback:
                current_params = parameter_space.vector_to_dict(current_vector)
                callback(iteration, current_params, current_cost)
            
            # Progress
            if verbose and iteration % 1000 == 0:
                acceptance_rate = n_accepted / (n_accepted + n_rejected) if (n_accepted + n_rejected) > 0 else 0
                print(f"Iteration {iteration}: best = {best_cost:.6f}, "
                      f"current = {current_cost:.6f}, T = {temperature:.2e}, "
                      f"acceptance = {acceptance_rate:.2f}")
        
        else:
            success = False
            message = f"Maximum iterations ({max_iterations}) reached"
        
        # Final result
        final_params = parameter_space.vector_to_dict(best_vector)
        
        return OptimizationResult(
            params=final_params,
            cost=float(best_cost),
            n_iterations=len(convergence_history),
            n_evaluations=cost_wrapper.n_evaluations,
            convergence_history=convergence_history,
            parameter_history=parameter_history,
            success=success,
            message=message,
            optimization_time=0.0
        )
    
    def _generate_neighbor(self,
                          key: jax.random.PRNGKey,
                          current_vector: jnp.ndarray,
                          temperature: float,
                          parameter_space) -> jnp.ndarray:
        """
        Generate neighbor solution using temperature-dependent perturbation.
        
        Args:
            key: Random key
            current_vector: Current solution vector
            temperature: Current temperature
            parameter_space: Parameter space for bounds
            
        Returns:
            Neighbor solution vector
        """
        lower, upper = parameter_space.get_bounds_vector()
        param_ranges = upper - lower
        
        # Temperature-dependent step size
        step_size = 0.1 * temperature / 1000.0  # Normalize temperature effect
        
        # Generate random perturbation
        perturbation = random.normal(key, current_vector.shape) * step_size * param_ranges
        
        # Apply perturbation
        neighbor = current_vector + perturbation
        
        # Clip to bounds
        neighbor = jnp.clip(neighbor, lower, upper)
        
        return neighbor


class AdaptiveSimulatedAnnealing(SimulatedAnnealing):
    """
    Adaptive Simulated Annealing with dynamic parameter adjustment.
    """
    
    def optimize(self,
                 cost_wrapper,
                 parameter_space,
                 initial_params: Dict[str, float],
                 key: jax.random.PRNGKey,
                 algorithm_params: Dict[str, Any],
                 max_iterations: int,
                 tolerance: float,
                 callback: Optional[Callable] = None,
                 verbose: bool = True) -> OptimizationResult:
        """
        Run adaptive simulated annealing with dynamic cooling schedule.
        """
        # Enhanced parameters for adaptive SA
        enhanced_params = algorithm_params.copy()
        enhanced_params.setdefault('adaptive_cooling', True)
        enhanced_params.setdefault('target_acceptance_rate', 0.44)  # Optimal for SA
        enhanced_params.setdefault('cooling_adjustment_factor', 0.1)
        
        return super().optimize(
            cost_wrapper, parameter_space, initial_params, key,
            enhanced_params, max_iterations, tolerance, callback, verbose
        )
    
    def _adaptive_cooling(self, temperature: float, acceptance_rate: float, 
                         target_rate: float, adjustment_factor: float) -> float:
        """
        Adapt cooling rate based on acceptance rate.
        
        Args:
            temperature: Current temperature
            acceptance_rate: Current acceptance rate
            target_rate: Target acceptance rate
            adjustment_factor: Adjustment strength
            
        Returns:
            Adjusted temperature
        """
        if acceptance_rate > target_rate:
            # Cool faster if accepting too much
            temperature *= (1 - adjustment_factor)
        elif acceptance_rate < target_rate * 0.5:
            # Cool slower if accepting too little
            temperature *= (1 + adjustment_factor)
        
        return temperature
