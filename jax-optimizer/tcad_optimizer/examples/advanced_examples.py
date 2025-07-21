"""
Advanced optimization examples for TCAD applications.
"""

import jax.numpy as jnp
import jax
import numpy as np
from typing import Dict, List, Callable
from tcad_optimizer import TCADOptimizer, ParameterSpace
from tcad_optimizer.core.cost_function import create_surrogate_cost_function
from tcad_optimizer.algorithms.hybrid import MultiObjectiveOptimizer
from tcad_optimizer.utils.logging import OptimizationLogger
from tcad_optimizer.utils.visualization import OptimizationVisualizer


class MockSurrogateModel:
    """
    Mock surrogate model for demonstration.
    
    In practice, this would be a trained ML model (neural network, 
    Gaussian process, etc.) that predicts device characteristics.
    """
    
    def __init__(self):
        # Simulate trained model parameters
        self.weights = {
            'current': jnp.array([2.5, -1.2, 0.8]),
            'voltage': jnp.array([0.3, 0.1, -0.4]),
            'power': jnp.array([1.8, 0.5, 1.1])
        }
    
    def predict(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Predict device characteristics from geometry parameters.
        
        Args:
            params: Geometry parameters
            
        Returns:
            Predicted device characteristics
        """
        # Normalize inputs
        width_norm = (params['width'] - 5.0) / 5.0
        height_norm = (params['height'] - 2.5) / 2.5
        doping_norm = (jnp.log10(params['doping']) - 16.5) / 1.5
        
        features = jnp.array([width_norm, height_norm, doping_norm])
        
        # Simulate nonlinear device physics
        predictions = {}
        
        # Current (mA)
        current_linear = jnp.dot(self.weights['current'], features)
        predictions['current'] = 10.0 + 5.0 * jnp.tanh(current_linear)
        
        # Voltage (V)
        voltage_linear = jnp.dot(self.weights['voltage'], features)
        predictions['voltage'] = 0.7 + 0.2 * jnp.tanh(voltage_linear)
        
        # Power consumption (mW)
        power_linear = jnp.dot(self.weights['power'], features)
        predictions['power'] = 5.0 + 3.0 * jnp.exp(0.5 * power_linear)
        
        return {k: float(v) for k, v in predictions.items()}


def surrogate_model_optimization():
    """
    Example using a surrogate model for optimization.
    """
    print("=== Surrogate Model Optimization Example ===\n")
    
    # Create surrogate model
    surrogate = MockSurrogateModel()
    
    # Define target specifications
    target_specs = {
        'current': 12.0,    # Target current: 12 mA
        'voltage': 0.75,    # Target voltage: 0.75 V
        'power': 6.0        # Target power: 6 mW
    }
    
    # Create cost function using surrogate
    cost_function = create_surrogate_cost_function(surrogate, target_specs)
    
    # Parameter space
    parameter_space = ParameterSpace({
        'width': (2.0, 8.0),
        'height': (1.0, 4.0),
        'doping': (5e15, 2e17)
    })
    
    print("Target specifications:")
    for spec, target in target_specs.items():
        print(f"  {spec}: {target}")
    
    # Optimize using hybrid approach for better global search
    optimizer = TCADOptimizer(
        cost_function=cost_function,
        parameter_space=parameter_space,
        algorithm='hybrid_de_adam',
        algorithm_params={
            'de_population': 40,
            'de_generations': 80,
            'adam_learning_rate': 0.02
        }
    )
    
    result = optimizer.optimize(
        max_iterations=300,
        tolerance=1e-8,
        verbose=True
    )
    
    print(f"\n=== Optimization Results ===")
    print(f"Final cost: {result.cost:.6f}")
    print(f"Optimal parameters:")
    for name, value in result.params.items():
        print(f"  {name}: {value:.4f}")
    
    # Validate with surrogate model
    predictions = surrogate.predict(result.params)
    print(f"\nPredicted characteristics:")
    for spec, predicted in predictions.items():
        target = target_specs[spec]
        error = abs(predicted - target) / target * 100
        print(f"  {spec}: {predicted:.4f} (target: {target}, error: {error:.1f}%)")
    
    return result, surrogate


def robust_optimization():
    """
    Example of robust optimization considering parameter uncertainties.
    """
    print("\n=== Robust Optimization Example ===\n")
    
    def robust_cost_function(params):
        """
        Cost function that considers parameter uncertainties.
        """
        # Nominal cost
        nominal_cost = example_cost_function(params)
        
        # Add uncertainty by sampling parameter variations
        key = jax.random.PRNGKey(42)
        n_samples = 20
        
        # Parameter uncertainties (±5%)
        uncertainty = 0.05
        costs = []
        
        for i in range(n_samples):
            key, subkey = jax.random.split(key)
            
            # Generate perturbed parameters
            perturbed_params = {}
            for name, value in params.items():
                noise = jax.random.normal(subkey, ()) * uncertainty * value
                perturbed_params[name] = value + noise
                # Split key for next parameter
                key, subkey = jax.random.split(key)
            
            # Evaluate cost for perturbed parameters
            perturbed_cost = example_cost_function(perturbed_params)
            costs.append(perturbed_cost)
        
        # Robust cost: mean + penalty for variance
        mean_cost = jnp.mean(jnp.array(costs))
        cost_variance = jnp.var(jnp.array(costs))
        robust_cost = mean_cost + 0.5 * cost_variance
        
        return float(robust_cost)
    
    parameter_space = ParameterSpace({
        'width': (1.0, 10.0),
        'height': (0.5, 5.0),
        'doping': (1e15, 1e18)
    })
    
    # Compare nominal vs robust optimization
    print("Running nominal optimization...")
    nominal_optimizer = TCADOptimizer(
        cost_function=example_cost_function,
        parameter_space=parameter_space,
        algorithm='adam'
    )
    nominal_result = nominal_optimizer.optimize(max_iterations=200, verbose=False)
    
    print("Running robust optimization...")
    robust_optimizer = TCADOptimizer(
        cost_function=robust_cost_function,
        parameter_space=parameter_space,
        algorithm='adam'
    )
    robust_result = robust_optimizer.optimize(max_iterations=200, verbose=False)
    
    print(f"\nComparison:")
    print(f"Nominal optimization - Cost: {nominal_result.cost:.6f}")
    print(f"Robust optimization  - Cost: {robust_result.cost:.6f}")
    
    return nominal_result, robust_result


def multi_objective_optimization():
    """
    Example of multi-objective optimization.
    """
    print("\n=== Multi-Objective Optimization Example ===\n")
    
    def performance_objective(params):
        """Maximize performance (minimize negative performance)."""
        width = params['width']
        height = params['height']
        doping = params['doping']
        
        # Performance metric (higher is better)
        performance = width * height * (doping / 1e16) ** 0.3
        return -performance  # Minimize negative performance
    
    def power_objective(params):
        """Minimize power consumption."""
        width = params['width']
        height = params['height']
        doping = params['doping']
        
        # Power increases with area and doping
        power = width * height * (doping / 1e15) ** 0.5
        return power
    
    def area_objective(params):
        """Minimize device area."""
        return params['width'] * params['height']
    
    # Define objectives
    objectives = [performance_objective, power_objective, area_objective]
    objective_names = ['Performance', 'Power', 'Area']
    
    parameter_space = ParameterSpace({
        'width': (1.0, 8.0),
        'height': (1.0, 6.0),
        'doping': (1e15, 1e17)
    })
    
    # Run multi-objective optimization
    mo_optimizer = MultiObjectiveOptimizer()
    key = jax.random.PRNGKey(42)
    
    result = mo_optimizer.optimize(
        cost_functions=objectives,
        parameter_space=parameter_space,
        key=key,
        algorithm_params={'population_size': 80},
        max_iterations=100,
        verbose=True
    )
    
    print(f"\nFound {len(result['pareto_solutions'])} Pareto optimal solutions")
    
    # Visualize Pareto front
    visualizer = OptimizationVisualizer()
    fig = visualizer.plot_pareto_front(
        result['pareto_objectives'],
        objective_names,
        title="Performance vs Power vs Area Trade-off",
        save_path="plots/pareto_front.png"
    )
    
    # Print some example solutions
    print(f"\nExample Pareto optimal solutions:")
    for i in range(min(5, len(result['pareto_solutions']))):
        params = result['pareto_solutions'][i]
        objectives_val = result['pareto_objectives'][i]
        print(f"Solution {i+1}:")
        print(f"  Parameters: width={params['width']:.2f}, height={params['height']:.2f}, doping={params['doping']:.2e}")
        print(f"  Objectives: performance={-objectives_val[0]:.2f}, power={objectives_val[1]:.2f}, area={objectives_val[2]:.2f}")
    
    return result


def constrained_optimization():
    """
    Example with constraints (device must meet certain specifications).
    """
    print("\n=== Constrained Optimization Example ===\n")
    
    def constrained_cost_function(params):
        """
        Cost function with penalty for constraint violations.
        """
        # Base cost
        base_cost = example_cost_function(params)
        
        # Constraints
        width = params['width']
        height = params['height']
        doping = params['doping']
        
        penalty = 0.0
        
        # Constraint 1: Aspect ratio must be between 0.5 and 2.0
        aspect_ratio = width / height
        if aspect_ratio < 0.5:
            penalty += 1000 * (0.5 - aspect_ratio) ** 2
        elif aspect_ratio > 2.0:
            penalty += 1000 * (aspect_ratio - 2.0) ** 2
        
        # Constraint 2: Minimum device area
        area = width * height
        min_area = 2.0
        if area < min_area:
            penalty += 1000 * (min_area - area) ** 2
        
        # Constraint 3: Doping uniformity (difference between regions)
        # This is a simplified constraint
        max_doping_variation = 0.1 * doping
        if doping > 5e16:  # High doping case
            penalty += 100 * jnp.maximum(0, doping - 5e16) / 1e16
        
        return base_cost + penalty
    
    parameter_space = ParameterSpace({
        'width': (1.0, 10.0),
        'height': (0.5, 5.0),
        'doping': (1e15, 1e18)
    })
    
    optimizer = TCADOptimizer(
        cost_function=constrained_cost_function,
        parameter_space=parameter_space,
        algorithm='differential_evolution',
        algorithm_params={'population_size': 60}
    )
    
    result = optimizer.optimize(
        max_iterations=200,
        tolerance=1e-6,
        verbose=True
    )
    
    # Check constraint satisfaction
    width = result.params['width']
    height = result.params['height']
    doping = result.params['doping']
    
    aspect_ratio = width / height
    area = width * height
    
    print(f"\n=== Constraint Check ===")
    print(f"Aspect ratio: {aspect_ratio:.3f} (should be 0.5-2.0)")
    print(f"Device area: {area:.3f} μm² (should be ≥ 2.0)")
    print(f"Doping level: {doping:.2e} cm⁻³")
    
    constraints_satisfied = (
        0.5 <= aspect_ratio <= 2.0 and
        area >= 2.0
    )
    print(f"All constraints satisfied: {constraints_satisfied}")
    
    return result


def example_cost_function(params):
    """Reuse the basic cost function from basic_optimization.py"""
    width = params['width']
    height = params['height'] 
    doping = params['doping']
    
    current_density = 500 * (width / height) * (doping / 1e16) ** 0.5
    current_error = (current_density - 1000) ** 2 / 1000 ** 2
    
    threshold_voltage = 0.5 + 0.3 * jnp.log(doping / 1e15) / jnp.log(10)
    voltage_error = (threshold_voltage - 0.7) ** 2 / 0.7 ** 2
    
    leakage_current = 1e-12 * width * doping / 1e15
    leakage_penalty = jnp.maximum(0, leakage_current - 1e-9) * 1e9
    
    total_cost = current_error + voltage_error + leakage_penalty
    return float(total_cost)


if __name__ == "__main__":
    import os
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    print("Running advanced TCAD optimization examples...\n")
    
    # Surrogate model optimization
    surrogate_result, surrogate_model = surrogate_model_optimization()
    
    # Robust optimization
    nominal_result, robust_result = robust_optimization()
    
    # Multi-objective optimization
    mo_result = multi_objective_optimization()
    
    # Constrained optimization
    constrained_result = constrained_optimization()
    
    print("\n=== All Advanced Examples Completed ===")
    print("Results and visualizations saved to logs/ and plots/ directories.")
