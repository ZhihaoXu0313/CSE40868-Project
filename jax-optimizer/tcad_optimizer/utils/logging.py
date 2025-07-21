"""
Logging utilities for optimization tracking.
"""

import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict


class OptimizationLogger:
    """
    Logger for tracking optimization progress and results.
    """
    
    def __init__(self, log_dir: str = "optimization_logs"):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.start_time = None
        self.current_run_id = None
        self.iteration_logs = []
        
    def start_run(self, run_id: str, algorithm: str, parameters: Dict[str, Any]):
        """
        Start a new optimization run.
        
        Args:
            run_id: Unique identifier for this run
            algorithm: Algorithm name
            parameters: Algorithm parameters
        """
        self.current_run_id = run_id
        self.start_time = time.time()
        self.iteration_logs = []
        
        # Save run metadata
        metadata = {
            'run_id': run_id,
            'algorithm': algorithm,
            'parameters': parameters,
            'start_time': self.start_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_file = self.log_dir / f"{run_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def log_iteration(self, 
                     iteration: int,
                     cost: float,
                     parameters: Dict[str, float],
                     additional_info: Optional[Dict[str, Any]] = None):
        """
        Log an optimization iteration.
        
        Args:
            iteration: Iteration number
            cost: Current cost value
            parameters: Current parameter values
            additional_info: Additional information to log
        """
        if self.current_run_id is None:
            raise ValueError("Must call start_run() before logging iterations")
        
        log_entry = {
            'iteration': iteration,
            'cost': float(cost),
            'parameters': {k: float(v) for k, v in parameters.items()},
            'timestamp': time.time() - self.start_time
        }
        
        if additional_info:
            log_entry['additional_info'] = additional_info
        
        self.iteration_logs.append(log_entry)
        
        # Save to CSV for easy analysis
        csv_file = self.log_dir / f"{self.current_run_id}_iterations.csv"
        file_exists = csv_file.exists()
        
        with open(csv_file, 'a', newline='') as f:
            if not file_exists:
                # Write header
                writer = csv.writer(f)
                header = ['iteration', 'cost', 'timestamp'] + list(parameters.keys())
                writer.writerow(header)
            
            # Write data
            writer = csv.writer(f)
            row = [iteration, cost, time.time() - self.start_time] + list(parameters.values())
            writer.writerow(row)
    
    def finish_run(self, result, save_full_log: bool = True):
        """
        Finish the current optimization run.
        
        Args:
            result: OptimizationResult object
            save_full_log: Whether to save complete iteration log
        """
        if self.current_run_id is None:
            return
        
        # Save final result
        result_dict = asdict(result)
        result_dict['total_time'] = time.time() - self.start_time
        
        result_file = self.log_dir / f"{self.current_run_id}_result.json"
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        # Save full iteration log if requested
        if save_full_log:
            log_file = self.log_dir / f"{self.current_run_id}_full_log.json"
            with open(log_file, 'w') as f:
                json.dump(self.iteration_logs, f, indent=2)
        
        print(f"Optimization logs saved to {self.log_dir}")
        
        # Reset for next run
        self.current_run_id = None
        self.start_time = None
        self.iteration_logs = []
    
    def load_run_results(self, run_id: str) -> Dict[str, Any]:
        """
        Load results from a previous run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Dictionary with run results
        """
        result_file = self.log_dir / f"{run_id}_result.json"
        
        if not result_file.exists():
            raise FileNotFoundError(f"No results found for run {run_id}")
        
        with open(result_file, 'r') as f:
            return json.load(f)
    
    def list_runs(self) -> List[str]:
        """
        List all available run IDs.
        
        Returns:
            List of run IDs
        """
        result_files = list(self.log_dir.glob("*_result.json"))
        run_ids = [f.stem.replace("_result", "") for f in result_files]
        return sorted(run_ids)
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """
        Compare results from multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            Comparison results
        """
        comparison = {
            'runs': {},
            'best_run': None,
            'best_cost': float('inf')
        }
        
        for run_id in run_ids:
            try:
                result = self.load_run_results(run_id)
                comparison['runs'][run_id] = {
                    'final_cost': result['cost'],
                    'n_iterations': result['n_iterations'],
                    'n_evaluations': result['n_evaluations'],
                    'success': result['success'],
                    'total_time': result.get('total_time', 0)
                }
                
                if result['cost'] < comparison['best_cost']:
                    comparison['best_cost'] = result['cost']
                    comparison['best_run'] = run_id
                    
            except FileNotFoundError:
                print(f"Warning: Results not found for run {run_id}")
        
        return comparison


def create_callback_logger(logger: OptimizationLogger) -> callable:
    """
    Create a callback function for optimization that logs to the given logger.
    
    Args:
        logger: OptimizationLogger instance
        
    Returns:
        Callback function
    """
    def callback(iteration: int, parameters: Dict[str, float], cost: float):
        logger.log_iteration(iteration, cost, parameters)
    
    return callback
