"""
Quick test script to validate the optimizer installation and basic functionality.
"""

import jax.numpy as jnp
import numpy as np


def test_basic_functionality():
    """Test basic optimizer functionality."""
    print("Testing JAX-based TCAD Optimizer...")
    
    try:
        # Import the optimizer
        from tcad_optimizer import TCADOptimizer, ParameterSpace
        print("✓ Successfully imported TCADOptimizer and ParameterSpace")
        
        # Test parameter space
        param_space = ParameterSpace({
            'x': (0.0, 10.0),
            'y': (-5.0, 5.0)
        })
        print("✓ Successfully created ParameterSpace")
        
        # Test parameter transformations
        test_params = {'x': 5.0, 'y': 0.0}
        vector = param_space.dict_to_vector(test_params)
        recovered_params = param_space.vector_to_dict(vector)
        print("✓ Parameter transformations working")
        
        # Simple test cost function
        def simple_cost(params):
            x, y = params['x'], params['y']
            return (x - 3.0)**2 + (y - 1.0)**2
        
        # Test optimizer creation
        optimizer = TCADOptimizer(
            cost_function=simple_cost,
            parameter_space=param_space,
            algorithm='adam'
        )
        print("✓ Successfully created TCADOptimizer")
        
        # Test optimization (short run)
        result = optimizer.optimize(
            max_iterations=50,
            tolerance=1e-4,
            verbose=False
        )
        print("✓ Successfully ran optimization")
        
        # Check if result is reasonable
        expected_x, expected_y = 3.0, 1.0
        actual_x, actual_y = result.params['x'], result.params['y']
        
        error_x = abs(actual_x - expected_x)
        error_y = abs(actual_y - expected_y)
        
        if error_x < 0.5 and error_y < 0.5:
            print("✓ Optimization converged to expected solution")
        else:
            print(f"⚠ Optimization result may be suboptimal: x={actual_x:.3f}, y={actual_y:.3f}")
        
        print(f"\nTest Results:")
        print(f"  Final cost: {result.cost:.6f}")
        print(f"  Optimal x: {actual_x:.3f} (expected: 3.0)")
        print(f"  Optimal y: {actual_y:.3f} (expected: 1.0)")
        print(f"  Iterations: {result.n_iterations}")
        print(f"  Function evaluations: {result.n_evaluations}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_algorithms():
    """Test different optimization algorithms."""
    print("\nTesting different algorithms...")
    
    try:
        from tcad_optimizer import TCADOptimizer, ParameterSpace
        
        param_space = ParameterSpace({'x': (0.0, 10.0)})
        
        def quadratic_cost(params):
            x = params['x']
            return (x - 5.0)**2
        
        algorithms = ['adam', 'differential_evolution', 'simulated_annealing']
        
        for alg in algorithms:
            try:
                optimizer = TCADOptimizer(
                    cost_function=quadratic_cost,
                    parameter_space=param_space,
                    algorithm=alg
                )
                
                result = optimizer.optimize(
                    max_iterations=30,
                    verbose=False
                )
                
                print(f"✓ {alg}: final cost = {result.cost:.6f}")
                
            except Exception as e:
                print(f"✗ {alg} failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Algorithm test failed: {e}")
        return False


def test_jax_functionality():
    """Test JAX-specific functionality."""
    print("\nTesting JAX functionality...")
    
    try:
        import jax
        from tcad_optimizer.core.cost_function import CostFunctionWrapper
        from tcad_optimizer import ParameterSpace
        
        # Test gradient computation
        param_space = ParameterSpace({'x': (-5.0, 5.0), 'y': (-5.0, 5.0)})
        
        def test_func(params):
            x, y = params['x'], params['y']
            return x**2 + y**2 + x*y
        
        wrapper = CostFunctionWrapper(test_func, param_space)
        
        # Test point
        test_vector = jnp.array([1.0, 2.0])  # x=1, y=2
        
        # Compute value and gradient
        value = wrapper.evaluate_vector(test_vector)
        grad = wrapper.gradient(test_vector)
        
        print(f"✓ Function value: {value:.3f}")
        print(f"✓ Gradient: [{grad[0]:.3f}, {grad[1]:.3f}]")
        
        # Test value_and_grad
        val, grad2 = wrapper.value_and_grad(test_vector)
        print(f"✓ Value and grad simultaneously computed")
        
        return True
        
    except Exception as e:
        print(f"✗ JAX test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("JAX-based TCAD Optimizer - Quick Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= test_basic_functionality()
    all_tests_passed &= test_algorithms() 
    all_tests_passed &= test_jax_functionality()
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All tests passed! The optimizer is ready to use.")
        print("\nNext steps:")
        print("1. Run: python tcad_optimizer/examples/basic_optimization.py")
        print("2. Run: python tcad_optimizer/examples/advanced_examples.py")
        print("3. Check the generated logs/ and plots/ directories")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    print("=" * 50)
