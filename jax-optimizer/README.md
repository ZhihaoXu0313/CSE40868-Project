# JAX-based TCAD Optimizer

A comprehensive optimization framework for TCAD applications using JAX, supporting multiple optimization algorithms including:
- Differential Evolution
- Simulated Annealing
- Advanced gradient-based algorithms (Adam, RMSprop, etc.)
- Hybrid approaches

## Features

- **JAX-based**: Leverages JAX's automatic differentiation for gradient-based optimization
- **Multiple Algorithms**: Supports both gradient-free and gradient-based methods
- **Flexible Parameter Handling**: Works with geometry parameters as dictionaries
- **Surrogate Model Support**: Compatible with surrogate models for cost function evaluation
- **Extensible**: Easy to add new optimization algorithms

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import jax.numpy as jnp
from tcad_optimizer import TCADOptimizer, ParameterSpace

# Define parameter space
param_space = ParameterSpace({
    'width': (1.0, 10.0),
    'height': (0.5, 5.0),
    'doping': (1e15, 1e18)
})

# Define cost function (using surrogate model)
def cost_function(params):
    # Your surrogate model evaluation here
    return calculate_cost_function(params)

# Create optimizer
optimizer = TCADOptimizer(
    cost_function=cost_function,
    parameter_space=param_space,
    algorithm='adam'  # or 'differential_evolution', 'annealing'
)

# Run optimization
result = optimizer.optimize(max_iterations=1000)
print(f"Optimal parameters: {result.params}")
print(f"Final cost: {result.cost}")
```

## Project Structure

```
tcad_optimizer/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── optimizer.py          # Main optimizer class
│   ├── parameter_space.py    # Parameter space definition
│   └── cost_function.py      # Cost function utilities
├── algorithms/
│   ├── __init__.py
│   ├── gradient_based.py     # Gradient-based algorithms
│   ├── differential_evolution.py
│   ├── simulated_annealing.py
│   └── hybrid.py            # Hybrid approaches
├── utils/
│   ├── __init__.py
│   ├── logging.py
│   └── visualization.py
└── examples/
    ├── basic_optimization.py
    └── advanced_examples.py
```
