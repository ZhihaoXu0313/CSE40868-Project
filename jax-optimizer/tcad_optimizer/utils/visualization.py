"""
Visualization utilities for optimization results.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path


class OptimizationVisualizer:
    """
    Visualization tools for optimization results and analysis.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_convergence(self, 
                        convergence_history: List[float],
                        title: str = "Optimization Convergence",
                        log_scale: bool = True,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot convergence history.
        
        Args:
            convergence_history: List of cost values over iterations
            title: Plot title
            log_scale: Whether to use log scale for y-axis
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = range(len(convergence_history))
        ax.plot(iterations, convergence_history, linewidth=2, color=self.colors[0])
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if log_scale and min(convergence_history) > 0:
            ax.set_yscale('log')
        
        # Add annotations for best value
        min_cost = min(convergence_history)
        min_idx = convergence_history.index(min_cost)
        ax.annotate(f'Best: {min_cost:.2e} at iteration {min_idx}',
                   xy=(min_idx, min_cost),
                   xytext=(0.7, 0.9),
                   textcoords='axes fraction',
                   arrowprops=dict(arrowstyle='->', color='red'),
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_parameter_evolution(self,
                               parameter_history: List[Dict[str, float]],
                               param_names: Optional[List[str]] = None,
                               title: str = "Parameter Evolution",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot evolution of parameters over iterations.
        
        Args:
            parameter_history: List of parameter dictionaries
            param_names: Specific parameters to plot (all if None)
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if not parameter_history:
            raise ValueError("Parameter history is empty")
        
        all_param_names = list(parameter_history[0].keys())
        if param_names is None:
            param_names = all_param_names
        
        n_params = len(param_names)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        iterations = range(len(parameter_history))
        
        for i, param_name in enumerate(param_names):
            values = [params[param_name] for params in parameter_history]
            
            ax = axes[i] if n_params > 1 else axes[0]
            ax.plot(iterations, values, linewidth=2, color=self.colors[i % len(self.colors)])
            ax.set_xlabel('Iteration')
            ax.set_ylabel(param_name)
            ax.set_title(f'{param_name} Evolution')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_parameter_correlation(self,
                                 parameter_history: List[Dict[str, float]],
                                 title: str = "Parameter Correlation Matrix",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation matrix of parameters.
        
        Args:
            parameter_history: List of parameter dictionaries
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        import pandas as pd
        
        # Convert to DataFrame
        df = pd.DataFrame(parameter_history)
        correlation_matrix = df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(correlation_matrix.columns)
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Correlation Coefficient')
        
        # Add correlation values as text
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black")
        
        ax.set_title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_pareto_front(self,
                         objectives: np.ndarray,
                         objective_names: List[str],
                         title: str = "Pareto Front",
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Pareto front for multi-objective optimization.
        
        Args:
            objectives: Array of objective values (n_solutions x n_objectives)
            objective_names: Names of objectives
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        n_objectives = objectives.shape[1]
        
        if n_objectives == 2:
            # 2D Pareto front
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(objectives[:, 0], objectives[:, 1], 
                      c=self.colors[0], alpha=0.7, s=50)
            ax.set_xlabel(objective_names[0])
            ax.set_ylabel(objective_names[1])
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
        elif n_objectives == 3:
            # 3D Pareto front
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
                      c=self.colors[0], alpha=0.7, s=50)
            ax.set_xlabel(objective_names[0])
            ax.set_ylabel(objective_names[1])
            ax.set_zlabel(objective_names[2])
            ax.set_title(title)
            
        else:
            # Parallel coordinates plot for higher dimensions
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Normalize objectives to [0, 1] for better visualization
            normalized_obj = (objectives - objectives.min(axis=0)) / (objectives.max(axis=0) - objectives.min(axis=0))
            
            for i, solution in enumerate(normalized_obj):
                ax.plot(range(n_objectives), solution, alpha=0.7, color=self.colors[0])
            
            ax.set_xticks(range(n_objectives))
            ax.set_xticklabels(objective_names, rotation=45, ha='right')
            ax.set_ylabel('Normalized Objective Value')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def compare_algorithms(self,
                          results: Dict[str, Any],
                          metric: str = 'convergence',
                          title: str = "Algorithm Comparison",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare multiple optimization algorithms.
        
        Args:
            results: Dictionary with algorithm names as keys and results as values
            metric: Metric to compare ('convergence', 'final_cost', 'time')
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if metric == 'convergence':
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for i, (alg_name, result) in enumerate(results.items()):
                convergence = result.get('convergence_history', [])
                if convergence:
                    iterations = range(len(convergence))
                    ax.plot(iterations, convergence, 
                           label=alg_name, linewidth=2, 
                           color=self.colors[i % len(self.colors)])
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Cost')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
        elif metric == 'final_cost':
            fig, ax = plt.subplots(figsize=(10, 6))
            
            algorithms = list(results.keys())
            costs = [results[alg].get('cost', float('inf')) for alg in algorithms]
            
            bars = ax.bar(algorithms, costs, color=self.colors[:len(algorithms)])
            ax.set_ylabel('Final Cost')
            ax.set_title(title)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, cost in zip(bars, costs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{cost:.2e}', ha='center', va='bottom')
        
        elif metric == 'time':
            fig, ax = plt.subplots(figsize=(10, 6))
            
            algorithms = list(results.keys())
            times = [results[alg].get('optimization_time', 0) for alg in algorithms]
            
            bars = ax.bar(algorithms, times, color=self.colors[:len(algorithms)])
            ax.set_ylabel('Optimization Time (s)')
            ax.set_title(title)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, time_val in zip(bars, times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{time_val:.1f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_optimization_dashboard(self,
                                    result,
                                    save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Create a comprehensive dashboard of optimization results.
        
        Args:
            result: OptimizationResult object
            save_dir: Directory to save plots
            
        Returns:
            Dictionary of figure objects
        """
        figures = {}
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
        
        # Convergence plot
        if result.convergence_history:
            conv_path = save_dir / "convergence.png" if save_dir else None
            figures['convergence'] = self.plot_convergence(
                result.convergence_history, 
                save_path=conv_path
            )
        
        # Parameter evolution
        if result.parameter_history:
            param_path = save_dir / "parameter_evolution.png" if save_dir else None
            figures['parameter_evolution'] = self.plot_parameter_evolution(
                result.parameter_history,
                save_path=param_path
            )
            
            # Parameter correlation
            if len(result.parameter_history) > 10:  # Only if enough data points
                corr_path = save_dir / "parameter_correlation.png" if save_dir else None
                figures['parameter_correlation'] = self.plot_parameter_correlation(
                    result.parameter_history,
                    save_path=corr_path
                )
        
        return figures
