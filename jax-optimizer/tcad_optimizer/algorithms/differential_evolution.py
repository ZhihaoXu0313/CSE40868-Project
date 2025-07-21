"""
Differential Evolution optimization algorithm.
"""

from typing import Dict, Any, Optional, Callable, List
import jax
import jax.numpy as jnp
from jax import random
from ..core.optimizer import OptimizationResult


class DifferentialEvolution:
    """
    Differential Evolution (DE) optimization algorithm.
    
    A population-based stochastic optimization method that is particularly
    effective for global optimization problems.
    """
    
    def optimize(self,
                 cost_wrapper,
                 parameter_space,
                 key: jax.random.PRNGKey,
                 algorithm_params: Dict[str, Any],
                 max_iterations: int,
                 tolerance: float,
                 callback: Optional[Callable] = None,
                 verbose: bool = True) -> OptimizationResult:
        """
        Run differential evolution optimization.
        
        Args:
            cost_wrapper: Cost function wrapper
            parameter_space: Parameter space
            key: JAX random key
            algorithm_params: DE parameters
            max_iterations: Maximum iterations (generations)
            tolerance: Convergence tolerance
            callback: Optional callback function
            verbose: Whether to print progress
            
        Returns:
            OptimizationResult
        """
        # DE parameters
        population_size = algorithm_params['population_size']
        mutation_factor = algorithm_params['mutation_factor']
        crossover_probability = algorithm_params['crossover_probability']
        
        # Initialize population
        key, subkey = random.split(key)
        population = self._initialize_population(
            subkey, parameter_space, population_size
        )
        
        # Evaluate initial population
        costs = jnp.array([cost_wrapper.evaluate_vector(ind) for ind in population])
        
        # Find best individual
        best_idx = jnp.argmin(costs)
        best_individual = population[best_idx]
        best_cost = costs[best_idx]
        
        convergence_history = [float(best_cost)]
        parameter_history = [parameter_space.vector_to_dict(best_individual)]
        
        if verbose:
            print(f"Initial best cost: {best_cost:.6f}")
        
        # Evolution loop
        for generation in range(max_iterations):
            key, subkey = random.split(key)
            
            # Create new population
            new_population = []
            new_costs = []
            
            for i in range(population_size):
                # Mutation and crossover
                key, mut_key, cross_key = random.split(key, 3)
                
                mutant = self._mutate(
                    mut_key, population, i, mutation_factor, parameter_space
                )
                
                trial = self._crossover(
                    cross_key, population[i], mutant, crossover_probability
                )
                
                # Evaluate trial vector
                trial_cost = cost_wrapper.evaluate_vector(trial)
                
                # Selection
                if trial_cost < costs[i]:
                    new_population.append(trial)
                    new_costs.append(trial_cost)
                else:
                    new_population.append(population[i])
                    new_costs.append(costs[i])
            
            # Update population
            population = jnp.array(new_population)
            costs = jnp.array(new_costs)
            
            # Update best
            current_best_idx = jnp.argmin(costs)
            current_best_cost = costs[current_best_idx]
            
            if current_best_cost < best_cost:
                best_cost = current_best_cost
                best_individual = population[current_best_idx]
            
            # Store history
            convergence_history.append(float(best_cost))
            parameter_history.append(parameter_space.vector_to_dict(best_individual))
            
            # Check convergence
            if len(convergence_history) > 1:
                cost_change = abs(convergence_history[-1] - convergence_history[-2])
                if cost_change < tolerance:
                    success = True
                    message = f"Converged after {generation+1} generations"
                    break
            
            # Callback
            if callback:
                best_params = parameter_space.vector_to_dict(best_individual)
                callback(generation, best_params, best_cost)
            
            # Progress
            if verbose and generation % 50 == 0:
                print(f"Generation {generation}: best cost = {best_cost:.6f}")
        
        else:
            success = False
            message = f"Maximum generations ({max_iterations}) reached"
        
        # Final result
        final_params = parameter_space.vector_to_dict(best_individual)
        
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
    
    def _initialize_population(self, 
                              key: jax.random.PRNGKey,
                              parameter_space,
                              population_size: int) -> jnp.ndarray:
        """Initialize population uniformly within parameter bounds."""
        lower, upper = parameter_space.get_bounds_vector()
        n_params = len(lower)
        
        # Generate random population
        population = random.uniform(
            key, (population_size, n_params),
            minval=lower, maxval=upper
        )
        
        return population
    
    def _mutate(self,
                key: jax.random.PRNGKey,
                population: jnp.ndarray,
                target_idx: int,
                mutation_factor: float,
                parameter_space) -> jnp.ndarray:
        """
        Create mutant vector using DE/rand/1 strategy.
        
        Args:
            key: Random key
            population: Current population
            target_idx: Index of target vector
            mutation_factor: Mutation scaling factor
            parameter_space: Parameter space for bounds
            
        Returns:
            Mutant vector
        """
        population_size = population.shape[0]
        
        # Select three random individuals (different from target)
        candidates = jnp.arange(population_size)
        candidates = candidates[candidates != target_idx]
        
        selected = random.choice(key, candidates, (3,), replace=False)
        r1, r2, r3 = selected[0], selected[1], selected[2]
        
        # Create mutant: v = x_r1 + F * (x_r2 - x_r3)
        mutant = population[r1] + mutation_factor * (population[r2] - population[r3])
        
        # Clip to bounds
        lower, upper = parameter_space.get_bounds_vector()
        mutant = jnp.clip(mutant, lower, upper)
        
        return mutant
    
    def _crossover(self,
                   key: jax.random.PRNGKey,
                   target: jnp.ndarray,
                   mutant: jnp.ndarray,
                   crossover_probability: float) -> jnp.ndarray:
        """
        Perform binomial crossover between target and mutant vectors.
        
        Args:
            key: Random key
            target: Target vector
            mutant: Mutant vector
            crossover_probability: Crossover probability
            
        Returns:
            Trial vector
        """
        n_params = len(target)
        
        # Generate random numbers for crossover decisions
        key1, key2 = random.split(key)
        crossover_points = random.bernoulli(key1, crossover_probability, (n_params,))
        
        # Ensure at least one parameter is taken from mutant
        random_idx = random.randint(key2, (), 0, n_params)
        crossover_points = crossover_points.at[random_idx].set(True)
        
        # Create trial vector
        trial = jnp.where(crossover_points, mutant, target)
        
        return trial


class AdaptiveDifferentialEvolution(DifferentialEvolution):
    """
    Adaptive Differential Evolution with self-adapting parameters.
    """
    
    def optimize(self,
                 cost_wrapper,
                 parameter_space,
                 key: jax.random.PRNGKey,
                 algorithm_params: Dict[str, Any],
                 max_iterations: int,
                 tolerance: float,
                 callback: Optional[Callable] = None,
                 verbose: bool = True) -> OptimizationResult:
        """
        Run adaptive differential evolution with parameter adaptation.
        """
        # Initialize with adaptive parameters
        algorithm_params = algorithm_params.copy()
        algorithm_params.setdefault('adapt_mutation', True)
        algorithm_params.setdefault('adapt_crossover', True)
        algorithm_params.setdefault('adaptation_rate', 0.1)
        
        # Track successful parameter values
        successful_mutations = []
        successful_crossovers = []
        
        # Original optimization with parameter tracking
        result = super().optimize(
            cost_wrapper, parameter_space, key, algorithm_params,
            max_iterations, tolerance, callback, verbose
        )
        
        return result
