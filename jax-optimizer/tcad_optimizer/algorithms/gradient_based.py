"""
Gradient-based optimization algorithms using JAX and Optax.
"""

from typing import Dict, Any, Optional, Callable, List
import jax
import jax.numpy as jnp
import optax
from ..core.optimizer import OptimizationResult


class GradientBasedOptimizer:
    """
    Gradient-based optimization using Optax optimizers.
    
    Supports Adam, RMSprop, SGD, and L-BFGS algorithms.
    """
    
    def optimize(self,
                 cost_wrapper,
                 parameter_space,
                 initial_params: Dict[str, float],
                 algorithm: str,
                 algorithm_params: Dict[str, Any],
                 max_iterations: int,
                 tolerance: float,
                 callback: Optional[Callable] = None,
                 verbose: bool = True) -> OptimizationResult:
        """
        Run gradient-based optimization.
        
        Args:
            cost_wrapper: Cost function wrapper
            parameter_space: Parameter space
            initial_params: Initial parameters
            algorithm: Algorithm name ('adam', 'rmsprop', 'sgd', 'lbfgs')
            algorithm_params: Algorithm parameters
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            callback: Optional callback function
            verbose: Whether to print progress
            
        Returns:
            OptimizationResult
        """
        # Convert initial params to vector
        initial_vector = parameter_space.dict_to_vector(initial_params)
        
        # Create optimizer
        if algorithm == 'adam':
            optimizer = optax.adam(
                learning_rate=algorithm_params['learning_rate'],
                b1=algorithm_params['beta1'],
                b2=algorithm_params['beta2'],
                eps=algorithm_params['eps']
            )
        elif algorithm == 'rmsprop':
            optimizer = optax.rmsprop(
                learning_rate=algorithm_params['learning_rate'],
                decay=algorithm_params['decay'],
                eps=algorithm_params['eps']
            )
        elif algorithm == 'sgd':
            optimizer = optax.sgd(
                learning_rate=algorithm_params['learning_rate'],
                momentum=algorithm_params['momentum']
            )
        elif algorithm == 'lbfgs':
            return self._optimize_lbfgs(
                cost_wrapper, parameter_space, initial_vector,
                algorithm_params, max_iterations, tolerance, callback, verbose
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Initialize optimizer state
        opt_state = optimizer.init(initial_vector)
        
        # Optimization loop
        current_vector = initial_vector
        convergence_history = []
        parameter_history = []
        
        for iteration in range(max_iterations):
            # Compute cost and gradient
            cost, grad = cost_wrapper.value_and_grad(current_vector)
            
            # Store history
            convergence_history.append(float(cost))
            current_params = parameter_space.vector_to_dict(current_vector)
            parameter_history.append(current_params.copy())
            
            # Check convergence
            if len(convergence_history) > 1:
                cost_change = abs(convergence_history[-1] - convergence_history[-2])
                if cost_change < tolerance:
                    success = True
                    message = f"Converged after {iteration+1} iterations (cost change < {tolerance})"
                    break
            
            # Update parameters
            updates, opt_state = optimizer.update(grad, opt_state, current_vector)
            current_vector = optax.apply_updates(current_vector, updates)
            
            # Project to bounds (soft constraint)
            current_vector = self._project_to_bounds(current_vector, parameter_space)
            
            # Callback
            if callback:
                callback(iteration, current_params, cost)
            
            # Progress
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: cost = {cost:.6f}")
        
        else:
            success = False
            message = f"Maximum iterations ({max_iterations}) reached"
        
        # Final result
        final_params = parameter_space.vector_to_dict(current_vector)
        final_cost = float(convergence_history[-1])
        
        return OptimizationResult(
            params=final_params,
            cost=final_cost,
            n_iterations=len(convergence_history),
            n_evaluations=cost_wrapper.n_evaluations,
            convergence_history=convergence_history,
            parameter_history=parameter_history,
            success=success,
            message=message,
            optimization_time=0.0  # Will be set by caller
        )
    
    def _project_to_bounds(self, vector: jnp.ndarray, parameter_space) -> jnp.ndarray:
        """Project parameter vector to satisfy bounds constraints."""
        lower, upper = parameter_space.get_bounds_vector()
        return jnp.clip(vector, lower, upper)
    
    def _optimize_lbfgs(self,
                       cost_wrapper,
                       parameter_space,
                       initial_vector: jnp.ndarray,
                       algorithm_params: Dict[str, Any],
                       max_iterations: int,
                       tolerance: float,
                       callback: Optional[Callable],
                       verbose: bool) -> OptimizationResult:
        """
        L-BFGS optimization implementation.
        
        This is a simplified L-BFGS implementation. For production use,
        consider using scipy.optimize.minimize with L-BFGS-B.
        """
        # L-BFGS parameters
        learning_rate = algorithm_params['learning_rate']
        history_size = algorithm_params['history_size']
        
        # History storage
        s_history = []  # Parameter differences
        y_history = []  # Gradient differences
        rho_history = []  # 1 / (y^T s)
        
        current_vector = initial_vector
        current_cost, current_grad = cost_wrapper.value_and_grad(current_vector)
        
        convergence_history = [float(current_cost)]
        parameter_history = [parameter_space.vector_to_dict(current_vector)]
        
        for iteration in range(max_iterations):
            # Compute search direction using L-BFGS two-loop recursion
            if len(s_history) == 0:
                # First iteration: use steepest descent
                search_direction = -current_grad
            else:
                search_direction = self._lbfgs_two_loop(
                    current_grad, s_history, y_history, rho_history
                )
            
            # Line search (simple backtracking)
            step_size = learning_rate
            new_vector = current_vector + step_size * search_direction
            new_vector = self._project_to_bounds(new_vector, parameter_space)
            
            # Evaluate new point
            new_cost, new_grad = cost_wrapper.value_and_grad(new_vector)
            
            # Simple Armijo condition for line search
            c1 = 1e-4
            if new_cost <= current_cost + c1 * step_size * jnp.dot(current_grad, search_direction):
                # Accept step
                s = new_vector - current_vector
                y = new_grad - current_grad
                
                # Update L-BFGS history
                if jnp.dot(y, s) > 1e-10:  # Curvature condition
                    if len(s_history) >= history_size:
                        s_history.pop(0)
                        y_history.pop(0)
                        rho_history.pop(0)
                    
                    s_history.append(s)
                    y_history.append(y)
                    rho_history.append(1.0 / jnp.dot(y, s))
                
                current_vector = new_vector
                current_cost = new_cost
                current_grad = new_grad
            else:
                # Reduce step size
                learning_rate *= 0.5
                if learning_rate < 1e-10:
                    break
            
            # Store history
            convergence_history.append(float(current_cost))
            current_params = parameter_space.vector_to_dict(current_vector)
            parameter_history.append(current_params.copy())
            
            # Check convergence
            if len(convergence_history) > 1:
                cost_change = abs(convergence_history[-1] - convergence_history[-2])
                if cost_change < tolerance:
                    success = True
                    message = f"Converged after {iteration+1} iterations"
                    break
            
            # Callback
            if callback:
                callback(iteration, current_params, current_cost)
            
            # Progress
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: cost = {current_cost:.6f}")
        
        else:
            success = False
            message = f"Maximum iterations ({max_iterations}) reached"
        
        final_params = parameter_space.vector_to_dict(current_vector)
        
        return OptimizationResult(
            params=final_params,
            cost=float(current_cost),
            n_iterations=len(convergence_history),
            n_evaluations=cost_wrapper.n_evaluations,
            convergence_history=convergence_history,
            parameter_history=parameter_history,
            success=success,
            message=message,
            optimization_time=0.0
        )
    
    def _lbfgs_two_loop(self, grad, s_history, y_history, rho_history):
        """L-BFGS two-loop recursion to compute search direction."""
        q = grad.copy()
        alpha = []
        
        # First loop (backward)
        for i in range(len(s_history) - 1, -1, -1):
            alpha_i = rho_history[i] * jnp.dot(s_history[i], q)
            q = q - alpha_i * y_history[i]
            alpha.append(alpha_i)
        
        alpha.reverse()
        
        # Initial Hessian approximation (identity)
        r = q
        
        # Second loop (forward)
        for i in range(len(s_history)):
            beta = rho_history[i] * jnp.dot(y_history[i], r)
            r = r + s_history[i] * (alpha[i] - beta)
        
        return -r
