"""
Hybrid optimization algorithms combining multiple approaches.
"""

from typing import Dict, Any, Optional, Callable, List
import jax
import jax.numpy as jnp
from jax import random
from ..core.optimizer import OptimizationResult
from .differential_evolution import DifferentialEvolution
from .gradient_based import GradientBasedOptimizer


class HybridOptimizer:
    """
    Hybrid optimization combining global and local search methods.
    
    Typically uses Differential Evolution for global exploration
    followed by gradient-based local optimization.
    """
    
    def __init__(self):
        self.de_optimizer = DifferentialEvolution()
        self.grad_optimizer = GradientBasedOptimizer()
    
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
        Run hybrid optimization.
        
        Args:
            cost_wrapper: Cost function wrapper
            parameter_space: Parameter space
            key: JAX random key
            algorithm_params: Hybrid parameters
            max_iterations: Maximum total iterations
            tolerance: Convergence tolerance
            callback: Optional callback function
            verbose: Whether to print progress
            
        Returns:
            OptimizationResult
        """
        # Hybrid parameters
        de_population = algorithm_params['de_population']
        de_generations = algorithm_params['de_generations']
        adam_learning_rate = algorithm_params['adam_learning_rate']
        
        # Allocate iterations
        de_iterations = min(de_generations, max_iterations // 2)
        grad_iterations = max_iterations - de_iterations
        
        if verbose:
            print(f"Hybrid optimization: {de_iterations} DE generations + {grad_iterations} gradient steps")
        
        # Phase 1: Differential Evolution for global exploration
        if verbose:
            print("\nPhase 1: Differential Evolution (Global Search)")
        
        de_params = {
            'population_size': de_population,
            'mutation_factor': 0.8,
            'crossover_probability': 0.9
        }
        
        de_result = self.de_optimizer.optimize(
            cost_wrapper, parameter_space, key, de_params,
            de_iterations, tolerance, callback, verbose
        )
        
        # Phase 2: Gradient-based optimization for local refinement
        if verbose:
            print(f"\nPhase 2: Adam Optimization (Local Refinement)")
            print(f"Starting from DE result: cost = {de_result.cost:.6f}")
        
        adam_params = {
            'learning_rate': adam_learning_rate,
            'beta1': 0.9,
            'beta2': 0.999,
            'eps': 1e-8
        }
        
        grad_result = self.grad_optimizer.optimize(
            cost_wrapper, parameter_space, de_result.params,
            'adam', adam_params, grad_iterations, tolerance,
            callback, verbose
        )
        
        # Combine results
        total_convergence_history = de_result.convergence_history + grad_result.convergence_history
        total_parameter_history = de_result.parameter_history + grad_result.parameter_history
        
        # Determine best result
        if grad_result.cost < de_result.cost:
            final_params = grad_result.params
            final_cost = grad_result.cost
            success = grad_result.success
            message = f"Hybrid optimization: DE + Adam. Final phase: {grad_result.message}"
        else:
            final_params = de_result.params
            final_cost = de_result.cost
            success = de_result.success
            message = f"Hybrid optimization: DE better than Adam. {de_result.message}"
        
        return OptimizationResult(
            params=final_params,
            cost=final_cost,
            n_iterations=de_result.n_iterations + grad_result.n_iterations,
            n_evaluations=cost_wrapper.n_evaluations,
            convergence_history=total_convergence_history,
            parameter_history=total_parameter_history,
            success=success,
            message=message,
            optimization_time=0.0
        )


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization using NSGA-II approach.
    
    For TCAD applications where multiple objectives need to be optimized
    simultaneously (e.g., performance vs. power consumption).
    """
    
    def __init__(self):
        pass
    
    def optimize(self,
                 cost_functions: List[Callable],
                 parameter_space,
                 key: jax.random.PRNGKey,
                 algorithm_params: Dict[str, Any],
                 max_iterations: int,
                 verbose: bool = True) -> Dict[str, Any]:
        """
        Run multi-objective optimization using NSGA-II.
        
        Args:
            cost_functions: List of objective functions
            parameter_space: Parameter space
            key: JAX random key
            algorithm_params: Algorithm parameters
            max_iterations: Maximum iterations
            verbose: Whether to print progress
            
        Returns:
            Dictionary with Pareto front results
        """
        population_size = algorithm_params.get('population_size', 100)
        
        # Initialize population
        key, subkey = random.split(key)
        population = self._initialize_population(subkey, parameter_space, population_size)
        
        # Evaluate objectives for initial population
        objectives = self._evaluate_population(population, cost_functions, parameter_space)
        
        pareto_fronts = []
        
        # Evolution loop
        for generation in range(max_iterations):
            # Non-dominated sorting
            fronts = self._non_dominated_sort(objectives)
            
            # Crowding distance assignment
            for front in fronts:
                self._crowding_distance_assignment(objectives[front])
            
            # Selection, crossover, and mutation
            key, subkey = random.split(key)
            new_population = self._create_offspring(
                subkey, population, fronts, parameter_space
            )
            
            # Evaluate new population
            new_objectives = self._evaluate_population(
                new_population, cost_functions, parameter_space
            )
            
            # Combine populations
            combined_population = jnp.concatenate([population, new_population])
            combined_objectives = jnp.concatenate([objectives, new_objectives])
            
            # Environmental selection
            population, objectives = self._environmental_selection(
                combined_population, combined_objectives, population_size
            )
            
            if verbose and generation % 50 == 0:
                print(f"Generation {generation}: Population size = {len(population)}")
        
        # Extract final Pareto front
        fronts = self._non_dominated_sort(objectives)
        pareto_front_indices = fronts[0]
        pareto_solutions = population[pareto_front_indices]
        pareto_objectives = objectives[pareto_front_indices]
        
        # Convert to parameter dictionaries
        pareto_params = [
            parameter_space.vector_to_dict(sol) for sol in pareto_solutions
        ]
        
        return {
            'pareto_solutions': pareto_params,
            'pareto_objectives': pareto_objectives,
            'all_solutions': [parameter_space.vector_to_dict(sol) for sol in population],
            'all_objectives': objectives
        }
    
    def _initialize_population(self, key, parameter_space, population_size):
        """Initialize random population."""
        lower, upper = parameter_space.get_bounds_vector()
        n_params = len(lower)
        
        return random.uniform(
            key, (population_size, n_params),
            minval=lower, maxval=upper
        )
    
    def _evaluate_population(self, population, cost_functions, parameter_space):
        """Evaluate all objectives for population."""
        n_objectives = len(cost_functions)
        objectives = jnp.zeros((len(population), n_objectives))
        
        for i, individual in enumerate(population):
            params = parameter_space.vector_to_dict(individual)
            for j, cost_func in enumerate(cost_functions):
                objectives = objectives.at[i, j].set(cost_func(params))
        
        return objectives
    
    def _non_dominated_sort(self, objectives):
        """Perform non-dominated sorting."""
        n_individuals = len(objectives)
        dominated_count = jnp.zeros(n_individuals)
        dominates = [[] for _ in range(n_individuals)]
        fronts = [[]]
        
        # Find domination relationships
        for i in range(n_individuals):
            for j in range(n_individuals):
                if i != j:
                    if self._dominates(objectives[i], objectives[j]):
                        dominates[i].append(j)
                    elif self._dominates(objectives[j], objectives[i]):
                        dominated_count = dominated_count.at[i].add(1)
            
            if dominated_count[i] == 0:
                fronts[0].append(i)
        
        # Build subsequent fronts
        front_idx = 0
        while len(fronts[front_idx]) > 0:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominates[i]:
                    dominated_count = dominated_count.at[j].add(-1)
                    if dominated_count[j] == 0:
                        next_front.append(j)
            
            if len(next_front) > 0:
                fronts.append(next_front)
            front_idx += 1
        
        return [jnp.array(front) for front in fronts if len(front) > 0]
    
    def _dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2 (minimization)."""
        return jnp.all(obj1 <= obj2) and jnp.any(obj1 < obj2)
    
    def _crowding_distance_assignment(self, objectives):
        """Assign crowding distance to solutions."""
        n_solutions = len(objectives)
        n_objectives = objectives.shape[1]
        distances = jnp.zeros(n_solutions)
        
        for m in range(n_objectives):
            # Sort by objective m
            sorted_indices = jnp.argsort(objectives[:, m])
            
            # Set boundary points to infinity
            distances = distances.at[sorted_indices[0]].set(jnp.inf)
            distances = distances.at[sorted_indices[-1]].set(jnp.inf)
            
            # Calculate distances for intermediate points
            obj_range = objectives[sorted_indices[-1], m] - objectives[sorted_indices[0], m]
            if obj_range > 0:
                for i in range(1, n_solutions - 1):
                    distance = (objectives[sorted_indices[i+1], m] - 
                              objectives[sorted_indices[i-1], m]) / obj_range
                    distances = distances.at[sorted_indices[i]].add(distance)
        
        return distances
    
    def _create_offspring(self, key, population, fronts, parameter_space):
        """Create offspring through selection, crossover, and mutation."""
        # Simplified implementation - just return mutated copies
        n_offspring = len(population)
        key, subkey = random.split(key)
        
        # Select parents (simplified tournament selection)
        parent_indices = random.choice(subkey, len(population), (n_offspring,))
        offspring = population[parent_indices]
        
        # Apply mutation
        lower, upper = parameter_space.get_bounds_vector()
        key, subkey = random.split(key)
        mutation_strength = 0.1
        mutations = random.normal(subkey, offspring.shape) * mutation_strength
        offspring = jnp.clip(offspring + mutations, lower, upper)
        
        return offspring
    
    def _environmental_selection(self, population, objectives, target_size):
        """Select individuals for next generation."""
        # Non-dominated sorting
        fronts = self._non_dominated_sort(objectives)
        
        selected_indices = []
        
        # Add complete fronts
        for front in fronts:
            if len(selected_indices) + len(front) <= target_size:
                selected_indices.extend(front)
            else:
                # Partial front selection based on crowding distance
                remaining = target_size - len(selected_indices)
                if remaining > 0:
                    distances = self._crowding_distance_assignment(objectives[front])
                    sorted_front = front[jnp.argsort(-distances)]  # Descending order
                    selected_indices.extend(sorted_front[:remaining])
                break
        
        selected_indices = jnp.array(selected_indices)
        return population[selected_indices], objectives[selected_indices]
