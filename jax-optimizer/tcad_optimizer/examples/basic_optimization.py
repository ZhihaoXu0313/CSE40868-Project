"""
Basic optimization example demonstrating TCAD optimizer usage.
"""

import jax.numpy as jnp
import numpy as np
from tcad_optimizer import TCADOptimizer, ParameterSpace
from tcad_optimizer.utils.logging import OptimizationLogger, create_callback_logger
from tcad_optimizer.utils.visualization import OptimizationVisualizer


def example_cost_function(params):
    """
    Example cost function simulating a TCAD optimization problem.
    
    This represents optimizing a device with geometry parameters
    to achieve target electrical characteristics.
    """
    # Extract parameters
    width = params['width']
    height = params['height'] 
    doping = params['doping']
    
    # Simulate device performance (example formulas)
    # In real TCAD, this would call a device simulator or surrogate model
    
    # Current density (target: 1000 A/cm²)
    current_density = 500 * (width / height) * (doping / 1e16) ** 0.5
    current_error = (current_density - 1000) ** 2 / 1000 ** 2
    
    # Threshold voltage (target: 0.7 V)  
    threshold_voltage = 0.5 + 0.3 * jnp.log(doping / 1e15) / jnp.log(10)
    voltage_error = (threshold_voltage - 0.7) ** 2 / 0.7 ** 2
    
    # Leakage current (minimize, target < 1e-9 A)
    leakage_current = 1e-12 * width * doping / 1e15
    leakage_penalty = jnp.maximum(0, leakage_current - 1e-9) * 1e9
    
    # Combined cost function
    total_cost = current_error + voltage_error + leakage_penalty
    
    return float(total_cost)


def run_basic_optimization():
    """Run a basic optimization example."""
    print("=== Basic TCAD Optimization Example ===\n")
    
    # Define parameter space
    parameter_space = ParameterSpace({
        'width': (1.0, 10.0),      # Device width in μm
        'height': (0.5, 5.0),      # Device height in μm  
        'doping': (1e15, 1e18)     # Doping concentration in cm⁻³
    })
    
    print("Parameter space:")
    for name, (min_val, max_val) in parameter_space.bounds.items():
        print(f"  {name}: [{min_val:.2e}, {max_val:.2e}] ({parameter_space.transforms[name]} scale)")
    
    # Create optimizer
    optimizer = TCADOptimizer(
        cost_function=example_cost_function,
        parameter_space=parameter_space,
        algorithm='adam',
        algorithm_params={'learning_rate': 0.05}
    )
    
    # Setup logging
    logger = OptimizationLogger("logs/basic_example")
    logger.start_run("basic_adam", "adam", optimizer.algorithm_params)
    callback = create_callback_logger(logger)
    
    # Run optimization
    result = optimizer.optimize(
        max_iterations=500,
        tolerance=1e-6,
        callback=callback,
        verbose=True
    )
    
    logger.finish_run(result)
    
    # Print results
    print(f"\n=== Optimization Results ===")
    print(f"Success: {result.success}")
    print(f"Final cost: {result.cost:.6f}")
    print(f"Iterations: {result.n_iterations}")
    print(f"Function evaluations: {result.n_evaluations}")
    print(f"Optimization time: {result.optimization_time:.2f} seconds")
    print(f"\nOptimal parameters:")
    for name, value in result.params.items():
        print(f"  {name}: {value:.4f}")
    
    # Validate solution
    print(f"\n=== Solution Validation ===")
    optimal_params = result.params
    
    # Calculate individual objectives
    width = optimal_params['width']
    height = optimal_params['height']
    doping = optimal_params['doping']
    
    current_density = 500 * (width / height) * (doping / 1e16) ** 0.5
    threshold_voltage = 0.5 + 0.3 * np.log(doping / 1e15) / np.log(10)
    leakage_current = 1e-12 * width * doping / 1e15
    
    print(f"Current density: {current_density:.1f} A/cm² (target: 1000)")
    print(f"Threshold voltage: {threshold_voltage:.3f} V (target: 0.7)")
    print(f"Leakage current: {leakage_current:.2e} A (target: < 1e-9)")
    
    # Create visualizations
    visualizer = OptimizationVisualizer()
    figures = visualizer.create_optimization_dashboard(result, "plots/basic_example")
    
    print(f"\nVisualization plots saved to plots/basic_example/")
    
    return result


def compare_algorithms():
    """Compare different optimization algorithms."""
    print("\n=== Algorithm Comparison ===\n")
    
    # Define parameter space
    parameter_space = ParameterSpace({
        'width': (1.0, 10.0),
        'height': (0.5, 5.0),
        'doping': (1e15, 1e18)
    })
    
    # Algorithms to compare
    algorithms = {
        'adam': {'learning_rate': 0.05},
        'differential_evolution': {'population_size': 30},
        'simulated_annealing': {'initial_temperature': 1000.0},
        'hybrid_de_adam': {'de_population': 20, 'de_generations': 50}
    }
    
    results = {}
    logger = OptimizationLogger("logs/comparison")
    
    for alg_name, alg_params in algorithms.items():
        print(f"Running {alg_name}...")
        
        optimizer = TCADOptimizer(
            cost_function=example_cost_function,
            parameter_space=parameter_space,
            algorithm=alg_name,
            algorithm_params=alg_params
        )
        
        logger.start_run(f"comparison_{alg_name}", alg_name, alg_params)
        
        result = optimizer.optimize(
            max_iterations=200,
            tolerance=1e-6,
            verbose=False
        )
        
        logger.finish_run(result)
        results[alg_name] = result
        
        print(f"  Final cost: {result.cost:.6f}")
        print(f"  Time: {result.optimization_time:.2f}s")
        print(f"  Evaluations: {result.n_evaluations}")
    
    # Create comparison visualizations
    visualizer = OptimizationVisualizer()
    
    # Convergence comparison
    convergence_data = {name: result for name, result in results.items()}
    fig_conv = visualizer.compare_algorithms(
        convergence_data, 
        metric='convergence',
        title="Algorithm Convergence Comparison",
        save_path="plots/algorithm_convergence.png"
    )
    
    # Final cost comparison
    fig_cost = visualizer.compare_algorithms(
        convergence_data,
        metric='final_cost', 
        title="Final Cost Comparison",
        save_path="plots/algorithm_costs.png"
    )
    
    # Time comparison
    fig_time = visualizer.compare_algorithms(
        convergence_data,
        metric='time',
        title="Optimization Time Comparison", 
        save_path="plots/algorithm_times.png"
    )
    
    # Print summary
    print(f"\n=== Comparison Summary ===")
    best_alg = min(results.keys(), key=lambda k: results[k].cost)
    print(f"Best algorithm: {best_alg}")
    print(f"Best cost: {results[best_alg].cost:.6f}")
    
    return results


def multi_start_example():
    """Demonstrate multi-start optimization."""
    print("\n=== Multi-Start Optimization Example ===\n")
    
    parameter_space = ParameterSpace({
        'width': (1.0, 10.0),
        'height': (0.5, 5.0),
        'doping': (1e15, 1e18)
    })
    
    optimizer = TCADOptimizer(
        cost_function=example_cost_function,
        parameter_space=parameter_space,
        algorithm='adam'
    )
    
    # Run multi-start optimization
    result = optimizer.multi_start_optimization(
        n_starts=5,
        max_iterations=200,
        tolerance=1e-6,
        verbose=True
    )
    
    print(f"\nBest result from multi-start:")
    print(f"Cost: {result.cost:.6f}")
    print(f"Parameters: {result.params}")
    
    return result


if __name__ == "__main__":
    # Run examples
    import os
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Basic optimization
    basic_result = run_basic_optimization()
    
    # Algorithm comparison
    comparison_results = compare_algorithms()
    
    # Multi-start optimization
    multistart_result = multi_start_example()
    
    print("\n=== All Examples Completed ===")
    print("Check the logs/ and plots/ directories for detailed results and visualizations.")
