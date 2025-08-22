"""
Deterministic Training Framework

Production-grade reproducibility system:
- Complete environment determinism
- Distributed training synchronization  
- Experiment tracking and versioning
- Hardware-agnostic reproducibility
- Statistical validation of reproducibility
"""

import os
import random
import hashlib
import logging
import warnings
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

import numpy as np
import torch
import torch.backends.cudnn as cudnn

logger = logging.getLogger(__name__)

class DeterministicTrainingManager:
    """Complete deterministic training environment management."""
    
    def __init__(self, base_seed: int = 42, strict_mode: bool = True):
        self.base_seed = base_seed
        self.strict_mode = strict_mode
        self.environment_state = {}
        self.reproducibility_log = []
        
    def setup_deterministic_environment(self, worker_id: int = 0) -> Dict[str, Any]:
        """
        Setup complete deterministic environment.
        
        Args:
            worker_id: Worker ID for distributed training
            
        Returns:
            Environment configuration dictionary
        """
        
        # Calculate worker-specific seed
        worker_seed = self.base_seed + worker_id
        
        # Python standard library
        random.seed(worker_seed)
        
        # NumPy
        np.random.seed(worker_seed)
        
        # PyTorch
        torch.manual_seed(worker_seed)
        torch.cuda.manual_seed(worker_seed)
        torch.cuda.manual_seed_all(worker_seed)
        
        # Environment variables for determinism
        os.environ['PYTHONHASHSEED'] = str(worker_seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # CuDNN settings for determinism
        if torch.cuda.is_available():
            cudnn.deterministic = True
            cudnn.benchmark = False
            cudnn.enabled = True
            
            if self.strict_mode:
                # Force deterministic algorithms
                torch.use_deterministic_algorithms(True, warn_only=False)
            else:
                torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Record environment state
        env_config = {
            'worker_id': worker_id,
            'worker_seed': worker_seed,
            'python_seed': worker_seed,
            'numpy_seed': worker_seed,
            'torch_seed': worker_seed,
            'cuda_seed': worker_seed,
            'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cudnn_deterministic': cudnn.deterministic if torch.cuda.is_available() else None,
            'cudnn_benchmark': cudnn.benchmark if torch.cuda.is_available() else None,
            'torch_deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        self.environment_state = env_config
        
        logger.info(f"Deterministic environment configured for worker {worker_id}")
        logger.info(f"Environment config: {env_config}")
        
        return env_config
    
    def create_reproducible_dataloader(self, dataset, batch_size: int = 32,
                                     num_workers: int = 0, **kwargs) -> torch.utils.data.DataLoader:
        """Create reproducible PyTorch DataLoader."""
        
        def worker_init_fn(worker_id):
            # Set deterministic seed for each worker
            worker_seed = self.base_seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
            generator=torch.Generator().manual_seed(self.base_seed),
            **kwargs
        )
    
    def validate_reproducibility(self, model, dataloader, num_validation_runs: int = 3) -> Dict[str, Any]:
        """
        Validate that training is truly deterministic.
        
        Args:
            model: PyTorch model to test
            dataloader: DataLoader for testing
            num_validation_runs: Number of runs to compare
            
        Returns:
            Validation results with statistics
        """
        
        validation_results = {
            'is_reproducible': True,
            'run_outputs': [],
            'differences': [],
            'max_difference': 0.0,
            'mean_difference': 0.0
        }
        
        model.eval()
        
        # Run model multiple times with same input
        for run_idx in range(num_validation_runs):
            # Reset environment
            self.setup_deterministic_environment(worker_id=0)
            
            # Reset model state (if applicable)
            if hasattr(model, 'reset_parameters'):
                model.reset_parameters()
            
            # Get first batch for testing
            for batch in dataloader:
                inputs, targets = batch
                
                with torch.no_grad():
                    outputs = model(inputs)
                    validation_results['run_outputs'].append(outputs.clone())
                break  # Only test first batch
        
        # Compare outputs between runs
        if len(validation_results['run_outputs']) >= 2:
            base_output = validation_results['run_outputs'][0]
            
            for i, output in enumerate(validation_results['run_outputs'][1:], 1):
                diff = torch.abs(output - base_output)
                max_diff = torch.max(diff).item()
                mean_diff = torch.mean(diff).item()
                
                validation_results['differences'].append({
                    'run_comparison': f"0_vs_{i}",
                    'max_difference': max_diff,
                    'mean_difference': mean_diff
                })
                
                validation_results['max_difference'] = max(
                    validation_results['max_difference'], max_diff
                )
                validation_results['mean_difference'] += mean_diff
            
            validation_results['mean_difference'] /= len(validation_results['differences'])
            
            # Check if reproducible (tolerance of 1e-6)
            tolerance = 1e-6
            validation_results['is_reproducible'] = validation_results['max_difference'] < tolerance
        
        self.reproducibility_log.append({
            'timestamp': torch.cuda.Event(enable_timing=True),
            'validation_results': validation_results
        })
        
        if validation_results['is_reproducible']:
            logger.info("✓ Reproducibility validation PASSED")
        else:
            logger.warning(f"✗ Reproducibility validation FAILED - Max diff: {validation_results['max_difference']}")
        
        return validation_results
    
    def generate_experiment_hash(self, config: Dict[str, Any]) -> str:
        """Generate unique hash for experiment configuration."""
        
        # Include environment state and config
        hash_data = {
            'config': config,
            'environment': self.environment_state,
            'base_seed': self.base_seed
        }
        
        # Create reproducible hash
        hash_string = json.dumps(hash_data, sort_keys=True, default=str)
        experiment_hash = hashlib.sha256(hash_string.encode()).hexdigest()[:16]
        
        return experiment_hash
    
    def save_reproducibility_report(self, save_path: Path) -> None:
        """Save detailed reproducibility report."""
        
        report = {
            'base_seed': self.base_seed,
            'strict_mode': self.strict_mode,
            'environment_state': self.environment_state,
            'reproducibility_log': self.reproducibility_log,
            'system_info': {
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
                'python_version': os.sys.version,
                'platform': os.name
            }
        }
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Reproducibility report saved to {save_path}")

class ExperimentTracker:
    """Advanced experiment tracking and versioning."""
    
    def __init__(self, experiment_name: str, base_dir: Path = Path("experiments")):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.experiment_dir = base_dir / experiment_name
        self.current_run_id = None
        self.metrics_log = []
        
        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
    
    def start_run(self, run_config: Dict[str, Any], run_id: Optional[str] = None) -> str:
        """Start new experiment run with configuration."""
        
        if run_id is None:
            # Generate run ID from timestamp and config hash
            import time
            timestamp = int(time.time())
            config_hash = hashlib.md5(str(run_config).encode()).hexdigest()[:8]
            run_id = f"run_{timestamp}_{config_hash}"
        
        self.current_run_id = run_id
        run_dir = self.experiment_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save run configuration
        config_path = run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(run_config, f, indent=2, default=str)
        
        # Initialize metrics log
        self.metrics_log = []
        
        logger.info(f"Started experiment run: {self.experiment_name}/{run_id}")
        return run_id
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log training metrics."""
        
        if self.current_run_id is None:
            raise ValueError("No active run. Call start_run() first.")
        
        log_entry = {
            'step': step,
            'timestamp': torch.cuda.Event(enable_timing=True),
            'metrics': metrics
        }
        
        self.metrics_log.append(log_entry)
        
        # Also log to file for persistence
        run_dir = self.experiment_dir / self.current_run_id
        metrics_file = run_dir / "metrics.jsonl"
        
        with open(metrics_file, 'a') as f:
            json.dump(log_entry, f, default=str)
            f.write('\n')
    
    def save_model_checkpoint(self, model, optimizer, step: int, 
                            additional_state: Dict[str, Any] = None) -> Path:
        """Save model checkpoint with versioning."""
        
        if self.current_run_id is None:
            raise ValueError("No active run. Call start_run() first.")
        
        run_dir = self.experiment_dir / self.current_run_id
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
        
        checkpoint_data = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'run_id': self.current_run_id,
            'experiment_name': self.experiment_name
        }
        
        if additional_state:
            checkpoint_data.update(additional_state)
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Also save as latest checkpoint
        latest_path = checkpoint_dir / "latest.pt"
        torch.save(checkpoint_data, latest_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Path, model, optimizer) -> Dict[str, Any]:
        """Load model checkpoint."""
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint_data['model_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        return checkpoint_data

class MLOpsMonitoring:
    """Production MLOps monitoring and alerting."""
    
    def __init__(self, monitoring_config: Dict[str, Any] = None):
        self.monitoring_config = monitoring_config or {}
        self.alerts = []
        self.performance_metrics = {}
        
    def monitor_training_progress(self, metrics: Dict[str, float], step: int) -> List[str]:
        """Monitor training progress and generate alerts."""
        
        alerts = []
        
        # Check for training instability
        if 'loss' in metrics:
            loss = metrics['loss']
            
            # NaN/Inf detection
            if np.isnan(loss) or np.isinf(loss):
                alerts.append(f"CRITICAL: Loss is {loss} at step {step}")
            
            # Exploding gradients (heuristic)
            elif loss > 1000:
                alerts.append(f"WARNING: Very high loss {loss:.2f} at step {step}")
        
        # Learning rate monitoring
        if 'learning_rate' in metrics:
            lr = metrics['learning_rate']
            if lr < 1e-8:
                alerts.append(f"WARNING: Very low learning rate {lr:.2e} at step {step}")
        
        # GPU memory monitoring
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memory_pct = memory_used / memory_total
            
            if memory_pct > 0.9:
                alerts.append(f"WARNING: High GPU memory usage {memory_pct:.1%} at step {step}")
        
        # Store alerts
        self.alerts.extend(alerts)
        
        return alerts
    
    def monitor_model_performance(self, model, validation_data) -> Dict[str, Any]:
        """Monitor model performance and detect drift."""
        
        model.eval()
        performance_stats = {
            'parameter_stats': {},
            'gradient_stats': {},
            'activation_stats': {}
        }
        
        # Parameter statistics
        for name, param in model.named_parameters():
            if param is not None:
                performance_stats['parameter_stats'][name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item(),
                    'norm': param.data.norm().item()
                }
        
        # Gradient statistics (if available)
        for name, param in model.named_parameters():
            if param.grad is not None:
                performance_stats['gradient_stats'][name] = {
                    'mean': param.grad.mean().item(),
                    'std': param.grad.std().item(),
                    'norm': param.grad.norm().item()
                }
        
        return performance_stats
    
    def generate_alert_summary(self) -> Dict[str, Any]:
        """Generate summary of all alerts."""
        
        alert_summary = {
            'total_alerts': len(self.alerts),
            'critical_alerts': len([a for a in self.alerts if 'CRITICAL' in a]),
            'warning_alerts': len([a for a in self.alerts if 'WARNING' in a]),
            'recent_alerts': self.alerts[-10:],  # Last 10 alerts
            'alert_types': {}
        }
        
        # Categorize alerts
        for alert in self.alerts:
            if 'loss' in alert.lower():
                alert_summary['alert_types'].setdefault('loss_issues', 0)
                alert_summary['alert_types']['loss_issues'] += 1
            elif 'memory' in alert.lower():
                alert_summary['alert_types'].setdefault('memory_issues', 0)
                alert_summary['alert_types']['memory_issues'] += 1
            elif 'learning_rate' in alert.lower():
                alert_summary['alert_types'].setdefault('lr_issues', 0)
                alert_summary['alert_types']['lr_issues'] += 1
        
        return alert_summary

class ProductionDeploymentManager:
    """Production deployment and model serving management."""
    
    def __init__(self, model_registry_path: Path = Path("model_registry")):
        self.model_registry_path = model_registry_path
        self.model_registry_path.mkdir(parents=True, exist_ok=True)
        self.deployed_models = {}
    
    def register_model(self, model, model_name: str, version: str,
                      metadata: Dict[str, Any] = None) -> Path:
        """Register model in production registry."""
        
        model_dir = self.model_registry_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        
        # Save metadata
        model_metadata = {
            'model_name': model_name,
            'version': version,
            'registration_time': torch.cuda.Event(enable_timing=True),
            'model_architecture': str(model),
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'metadata': metadata or {}
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2, default=str)
        
        logger.info(f"Registered model: {model_name} v{version}")
        return model_path
    
    def load_production_model(self, model_class, model_name: str, 
                            version: str = "latest") -> torch.nn.Module:
        """Load model from production registry."""
        
        if version == "latest":
            # Find latest version
            model_base_dir = self.model_registry_path / model_name
            if not model_base_dir.exists():
                raise ValueError(f"Model {model_name} not found in registry")
            
            version_dirs = [d for d in model_base_dir.iterdir() if d.is_dir()]
            if not version_dirs:
                raise ValueError(f"No versions found for model {model_name}")
            
            # Sort by modification time and get latest
            latest_version_dir = max(version_dirs, key=lambda x: x.stat().st_mtime)
            version = latest_version_dir.name
        
        model_dir = self.model_registry_path / model_name / version
        model_path = model_dir / "model.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        model = model_class()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        # Load metadata
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded model {model_name} v{version} - {metadata.get('parameter_count', 'Unknown')} parameters")
        
        return model
    
    def create_model_api(self, model, preprocessing_fn: Optional[callable] = None,
                        postprocessing_fn: Optional[callable] = None) -> callable:
        """Create production API wrapper for model."""
        
        def model_api(input_data):
            """Production model API endpoint."""
            
            try:
                # Preprocessing
                if preprocessing_fn:
                    processed_input = preprocessing_fn(input_data)
                else:
                    processed_input = input_data
                
                # Model inference
                model.eval()
                with torch.no_grad():
                    output = model(processed_input)
                
                # Postprocessing
                if postprocessing_fn:
                    final_output = postprocessing_fn(output)
                else:
                    final_output = output
                
                return {
                    'prediction': final_output,
                    'status': 'success',
                    'model_version': getattr(model, 'version', 'unknown')
                }
                
            except Exception as e:
                logger.error(f"Model API error: {str(e)}")
                return {
                    'prediction': None,
                    'status': 'error',
                    'error_message': str(e)
                }
        
        return model_api