"""
Bayesian Optimization wrapper for H2-Gym environments using Ray Tune
"""

import os
import yaml
import multiprocessing
import pickle
import json
import time
from datetime import datetime
from pathlib import Path
from random import randint
from typing import Dict, Optional, Union, Callable, Any, List, Tuple

import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.air import session

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from hydro_bo.envs.shipping.version2.utils import calculate_capex_opex


class BayesianOptimizer:
    """
    Ray-based Bayesian Optimization wrapper for shipping environment
    Uses context manager pattern to configure hyperparameters
    """

    def __init__(
        self,
        tmp_dir: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        resume_from_checkpoint: bool = False,
    ):
        """
        Initialize the Bayesian Optimizer

        Args:
            tmp_dir: Temporary directory for results
            checkpoint_dir: Directory for saving/loading checkpoints
            resume_from_checkpoint: Whether to attempt resuming from latest checkpoint
        """
        # Set up temporary directories
        if tmp_dir is None:
            self.tmp_dir = Path(__file__).parent.parent.parent / "tmp"
        else:
            self.tmp_dir = Path(tmp_dir)

        self.bayesopt_dir = self.tmp_dir / "bayesopt"
        self.ray_results_dir = self.tmp_dir / "ray_results"

        # Set up checkpoint directory
        if checkpoint_dir is None:
            self.checkpoint_dir = self.tmp_dir / "checkpoints"
        else:
            self.checkpoint_dir = Path(checkpoint_dir)

        # Create directories
        self.bayesopt_dir.mkdir(parents=True, exist_ok=True)
        self.ray_results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Configuration parameters (set via context manager)
        self._config = {
            "env_version": "shipping",
            "vector_type": "LH2",
            "renewables_type": "wind",
            "weather_file": "CoastalChile_15-20_Wind.csv",
            "conversion_trains": 10,
            "planning_results_file": None,
            "scale": (0.5, 2.0),  # (lower_multiplier, upper_multiplier)
            "use_planning_bounds": False,
            "search_space": None,  # Will be set in __exit__
            "solver": "gurobi",
        }

        # Default search space (will be created lazily to avoid import-time issues)
        self._default_search_space = None

        # Ray setup
        self._ray_initialized = False
        self.results = None
        self._configured = False

        # Checkpoint management
        self.resume_from_checkpoint = resume_from_checkpoint
        self._checkpoint_state = None
        self._trial_history = []

        # Resource management
        self._resource_config = {
            "num_cpus": 62,
            "num_gpus": 0,
            "resources_per_trial": {"cpu": 1, "memory": None, "gpu": 0},
            "max_concurrent_trials": 62,
        }

        # Loaded results management
        self._loaded_results = None
        self._loaded_experiment_name = None

    def _get_default_search_space(self) -> Dict[str, Any]:
        """Get default search space, creating it if necessary"""
        if self._default_search_space is None:
            self._default_search_space = {
                "compression_capacity": tune.uniform(447.2761, 894.5522),
                "electrolyser_capacity": tune.uniform(9903.6524, 19807.3048),
                "fuelcell_capacity": tune.uniform(1010.3651, 2020.7302),
                "hydrogen_storage_capacity": tune.uniform(219639.3132, 439278.6263),
                "renewable_energy_capacity": tune.uniform(984.2415, 1968.4830),
                "vector_storage_capacity": tune.uniform(20.3036, 40.6072),
            }
        return self._default_search_space

    def _generate_bounds_from_file(
        self, planning_file: str, scale: tuple
    ) -> Dict[str, Any]:
        """
        Generate search space bounds from planning results file

        Args:
            planning_file: Path to planning results YAML file
            scale: Tuple of (lower_multiplier, upper_multiplier) for bounds

        Returns:
            Dictionary of search space bounds
        """
        try:
            # Load planning results
            planning_path = Path(planning_file)
            if not planning_path.is_absolute():
                # Try relative to tmp/planning directory
                planning_path = self.tmp_dir / "planning" / planning_file

            with open(planning_path, "r") as f:
                planning_results = yaml.safe_load(f)

            # Extract capacity values from planning results
            base_values = {
                "compression_capacity": planning_results.get(
                    "compression_capacity", 671.0
                ),
                "electrolyser_capacity": planning_results.get(
                    "electrolyser_capacity", 14855.0
                ),
                "fuelcell_capacity": planning_results.get("fuelcell_capacity", 1515.0),
                "hydrogen_storage_capacity": planning_results.get(
                    "hydrogen_storage_capacity", 329459.0
                ),
                "renewable_energy_capacity": planning_results.get(
                    "renewable_energy_capacity", 1476.0
                ),
                "vector_storage_capacity": planning_results.get(
                    "vector_storage_capacity", 30.5
                ),
            }

            # Generate bounds using scale tuple
            lower_mult, upper_mult = scale
            search_space = {}

            for param, base_value in base_values.items():
                lower_bound = base_value * lower_mult
                upper_bound = base_value * upper_mult
                search_space[param] = tune.uniform(lower_bound, upper_bound)

            return search_space

        except Exception as e:
            raise Exception(
                f"Failed to generate bounds from planning file {planning_file}: {e}"
            )

    def get_config(self) -> Dict:
        """
        Get current configuration

        Returns:
            Current configuration dictionary
        """
        if not self._configured:
            raise ValueError("Optimizer not configured. Use context manager first.")
        return self._config.copy()

    def save_checkpoint(self, checkpoint_name: Optional[str] = None) -> str:
        """
        Save the current optimization state to a checkpoint file

        Args:
            checkpoint_name: Optional name for the checkpoint

        Returns:
            Path to the saved checkpoint file
        """
        if self.results is None:
            raise ValueError(
                "No optimization results to checkpoint. Run optimize() first."
            )

        # Generate checkpoint name with timestamp
        if checkpoint_name is None:
            checkpoint_name = "checkpoint"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}_{timestamp}.pkl"

        # Collect checkpoint data
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "config": self._config.copy(),
            "completed_trials": len(self.results.dataframe()),
            "best_score": self.results.get_best_trial("score", "max").last_result[
                "score"
            ],
            "search_space": self._config["search_space"],
            "trial_results": self.results.dataframe().to_dict(),
            "algorithm_state": None,  # OptunaSearch state is reconstructed from trial_results during resume
            "resource_config": self._resource_config.copy(),
        }

        # Save checkpoint
        with open(checkpoint_file, "wb") as f:
            pickle.dump(checkpoint_data, f)

        # Also save as "latest"
        latest_file = self.checkpoint_dir / "latest.pkl"
        with open(latest_file, "wb") as f:
            pickle.dump(checkpoint_data, f)

        return str(checkpoint_file)

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load a previous optimization state from checkpoint

        Args:
            checkpoint_path: Path to the checkpoint file or folder containing checkpoints

        Returns:
            Dictionary containing checkpoint state
        """
        checkpoint_path = Path(checkpoint_path)

        # Check if it's a directory
        if checkpoint_path.is_dir():
            # Find all .pkl files in the directory
            pkl_files = list(checkpoint_path.glob("*.pkl"))

            if not pkl_files:
                raise FileNotFoundError(
                    f"No checkpoint files found in directory: {checkpoint_path}"
                )

            # Sort by modification time and get the most recent
            # Exclude 'latest.pkl' from the search to get the actual timestamped checkpoint
            timestamped_files = [f for f in pkl_files if f.name != "latest.pkl"]

            if timestamped_files:
                # Get the most recent timestamped checkpoint
                checkpoint_file = max(
                    timestamped_files, key=lambda f: f.stat().st_mtime
                )
            else:
                # If no timestamped files, try latest.pkl
                latest_file = checkpoint_path / "latest.pkl"
                if latest_file.exists():
                    checkpoint_file = latest_file
                else:
                    raise FileNotFoundError(
                        f"No valid checkpoint files found in directory: {checkpoint_path}"
                    )

            print(f"Loading checkpoint from directory: {checkpoint_file.name}")
            checkpoint_path = checkpoint_file

        # Handle file path
        elif not checkpoint_path.exists():
            # Try in checkpoint directory
            checkpoint_path = self.checkpoint_dir / checkpoint_path.name
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with open(checkpoint_path, "rb") as f:
            self._checkpoint_state = pickle.load(f)

        return self._checkpoint_state

    def list_checkpoints(self, checkpoint_dir: Optional[str] = None) -> List[Dict]:
        """
        List all available checkpoints in a directory

        Args:
            checkpoint_dir: Directory to search for checkpoints (default: self.checkpoint_dir)

        Returns:
            List of dictionaries with checkpoint information
        """
        if checkpoint_dir is None:
            search_dir = self.checkpoint_dir
        else:
            search_dir = Path(checkpoint_dir)

        if not search_dir.exists():
            return []

        checkpoints = []
        for pkl_file in search_dir.glob("*.pkl"):
            try:
                file_info = {
                    "name": pkl_file.name,
                    "path": str(pkl_file),
                    "size": pkl_file.stat().st_size,
                    "modified": datetime.fromtimestamp(
                        pkl_file.stat().st_mtime
                    ).isoformat(),
                    "is_latest": pkl_file.name == "latest.pkl",
                }
                checkpoints.append(file_info)
            except Exception as e:
                print(f"Error reading checkpoint {pkl_file}: {e}")

        # Sort by modification time, newest first
        checkpoints.sort(key=lambda x: x["modified"], reverse=True)
        return checkpoints

    def resume_optimization(
        self, checkpoint_path: str, additional_samples: int, n_cores=None, num_gpus=None
    ) -> ray.tune.ExperimentAnalysis:
        """
        Resume optimization from a checkpoint with additional samples

        Args:
            checkpoint_path: Path to checkpoint file or folder containing checkpoints
            additional_samples: Number of additional trials to run

        Returns:
            Ray Tune ExperimentAnalysis object with combined results
        """
        # Load checkpoint (now handles both files and directories)
        checkpoint_state = self.load_checkpoint(checkpoint_path)

        # Restore configuration
        self._config = checkpoint_state["config"]
        self._resource_config = checkpoint_state.get(
            "resource_config", self._resource_config
        )
        self._configured = True

        # Restore previous trial results for warm-start
        previous_trials = checkpoint_state.get("trial_results", {})
        print(f"Resuming from {checkpoint_state['completed_trials']} trials")
        print(f"Best score so far: {checkpoint_state['best_score']}")

        # Initialize Ray if not already done
        if not self._ray_initialized:
            ray_init_config = {
                "num_cpus": 62,
                "num_gpus": 0,
            }
            ray.init(**ray_init_config)
            self._ray_initialized = True

        # Create OptunaSearch with previous results for warm-start
        from optuna import create_study, trial as optuna_trial
        import pandas as pd

        # Create an Optuna study
        study = create_study(direction="maximise")

        # Restore previous trials if available
        if previous_trials and len(previous_trials.get("score", [])) > 0:
            # Convert trial results back to DataFrame
            df = pd.DataFrame(previous_trials)

            # Add each previous trial to the study
            for idx, row in df.iterrows():
                # Extract parameter values
                params = {}
                for col in df.columns:
                    if col.startswith("config/"):
                        param_name = col.replace("config/", "")
                        params[param_name] = row[col]

                # Create a trial and tell Optuna about it
                trial = study.ask()
                for param_name, param_value in params.items():
                    # Suggest the historical value
                    trial._suggest(param_name, param_value)

                # Tell the study about the result
                study.tell(trial, row["score"])

            print(f"Restored {len(df)} previous trials to Optuna study")

        # Create OptunaSearch with the warm-started study
        algo = OptunaSearch(metric="score", mode="max", study=study)

        # Set concurrency limit
        max_concurrent = self._resource_config["max_concurrent_trials"]
        algo = ConcurrencyLimiter(algo, max_concurrent=62)

        # Setup scheduler
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            metric="score",
            mode="max",
            max_t=100,
            grace_period=1,
            reduction_factor=3,
        )

        # Run additional optimization with warm-started search
        experiment_name = f"resumed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = tune.run(
            self._objective_function,
            name=experiment_name,
            search_alg=algo,
            scheduler=scheduler,
            config=self._config["search_space"],
            num_samples=additional_samples,
            resources_per_trial={"cpu": 1},
            storage_path=str(self.ray_results_dir),
        )

        return self.results

    def setup_distributed(
        self,
        n_cores: Optional[int] = None,
        memory_per_trial: str = "2GB",
        gpu_per_trial: float = 0.0,
    ) -> Dict:
        """
        Configure distributed computing resources

        Args:
            n_cores: Number of CPU cores to use (default: all available cores minus 2)
            memory_per_trial: Memory allocation per trial (e.g., "2GB", "512MB")
            gpu_per_trial: GPU allocation per trial (fractional GPUs supported)

        Returns:
            Resource configuration dictionary
        """
        # Auto-detect cores if not specified
        if n_cores is None:
            total_cores = (62,)
            # Use all cores minus 2, but at least 1
            n_cores = max(1, total_cores - 2)
            print(
                f"Auto-detected {total_cores} CPU cores, using {n_cores} for optimization"
            )

        # Parse memory string
        memory_bytes = self._parse_memory_string(memory_per_trial)

        # Update resource configuration
        self._resource_config = {
            "num_cpus": 62,
            "num_gpus": int(gpu_per_trial * n_cores) if gpu_per_trial > 0 else 0,
            "resources_per_trial": {
                "cpu": 1,
                "memory": memory_bytes,
                "gpu": gpu_per_trial,
            },
            "max_concurrent_trials": 62,
        }

        return self._resource_config.copy()

    def get_resource_usage(self) -> Dict:
        """
        Get current resource utilization during optimization

        Returns:
            Dictionary with resource usage statistics
        """
        if not self._ray_initialized or self.results is None:
            return {
                "active_trials": 0,
                "queued_trials": 0,
                "completed_trials": 0,
                "cpu_usage": {"used": 0, "total": 62},
                "memory_usage": {"used_gb": 0, "total_gb": 0},
                "estimated_time_remaining": "N/A",
            }

        # Get trial information
        df = self.results.dataframe() if self.results else pd.DataFrame()
        completed_trials = len(df)

        # Estimate resource usage (simplified)
        return {
            "active_trials": min(
                self._resource_config["max_concurrent_trials"],
                self._config.get("num_samples", 0) - completed_trials,
            ),
            "queued_trials": max(
                0,
                self._config.get("num_samples", 0)
                - completed_trials
                - self._resource_config["max_concurrent_trials"],
            ),
            "completed_trials": completed_trials,
            "cpu_usage": {
                "used": min(
                    self._resource_config["max_concurrent_trials"],
                    self._config.get("num_samples", 0) - completed_trials,
                ),
                "total": 62,
            },
            "memory_usage": {
                "used_gb": 0,
                "total_gb": 0,
            },  # TODO: Implement actual memory tracking
            "estimated_time_remaining": self._estimate_time_remaining(completed_trials),
        }

    def _parse_memory_string(self, memory_str: str) -> int:
        """Parse memory string to bytes"""
        memory_str = memory_str.upper()
        if memory_str.endswith("GB"):
            return int(float(memory_str[:-2]) * 1024 * 1024 * 1024)
        elif memory_str.endswith("MB"):
            return int(float(memory_str[:-2]) * 1024 * 1024)
        elif memory_str.endswith("KB"):
            return int(float(memory_str[:-2]) * 1024)
        else:
            return int(memory_str)

    def _estimate_time_remaining(self, completed_trials: int) -> str:
        """Estimate remaining time based on completed trials"""
        if completed_trials == 0:
            return "N/A"

        total_trials = self._config.get("num_samples", 0)
        if completed_trials >= total_trials:
            return "0m"

        # Simple estimation (would need actual timing data for accuracy)
        avg_time_per_trial = 60  # seconds, placeholder
        remaining_trials = total_trials - completed_trials
        remaining_seconds = (
            remaining_trials
            * avg_time_per_trial
            / self._resource_config["max_concurrent_trials"]
        )

        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)

        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"

    def _objective_function(self, config: Dict) -> None:
        """
        Objective function for Ray Tune optimization

        Args:
            config: Dictionary of hyperparameters from Ray Tune
        """
        if not self._configured:
            raise ValueError("Optimizer not configured. Use context manager first.")

        # Extract parameters
        args = [
            config["compression_capacity"],
            config["electrolyser_capacity"],
            config["fuelcell_capacity"],
            config["hydrogen_storage_capacity"],
            config["renewable_energy_capacity"],
            config["vector_storage_capacity"],
        ]

        # Generate unique ID for this trial
        trial_id = randint(0, 1000000)

        # Calculate CAPEX/OPEX parameters
        results = calculate_capex_opex(
            renewables=self._config["renewables_type"],
            vector=self._config["vector_type"],
            compression_capacity=args[0],
            electrolyser_capacity=args[1],
            fuelcell_capacity=args[2],
            conversion_trains_number=self._config["conversion_trains"],
            hydrogen_storage_capacity=args[3],
            renewable_energy_capacity=args[4],
            vector_storage_capacity=args[5],
        )

        # Save planning parameters
        planning_file = self.bayesopt_dir / f"{trial_id}.yml"
        with open(planning_file, "w") as yml_file:
            yaml.dump(results, yml_file)

        # Run shipping environment
        from hydro_bo.envs import ShippingEnv

        environment = ShippingEnv(self._config["env_version"])

        with environment as env:
            env["vector"] = self._config["vector_type"]
            env["fast"]["data_folder"] = "version2"
            env["fast"]["solver"] = self._config["solver"]
            env["fast"]["planning_model"] = str(planning_file)
            env["weather_data"]["weather_file"] = self._config["weather_file"]

        # Execute environment and collect rewards
        total_reward = 0
        total_tonnes = 0
        while True:
            observation, reward, done, _ = environment.step(action=None, verbose=False)
            total_reward += reward
            total_tonnes += observation["total_tonnes"]
            if done:
                break

        # Report result to Ray (negative because Ray minimizes)
        session.report({"score": total_reward / total_tonnes, "trial_id": trial_id})

    def optimize(
        self,
        num_samples: int = 50,
        n_cores: Optional[int] = None,
        experiment_name: str = "shipping_bayesopt",
        ray_config: Optional[Dict] = None,
        enable_checkpointing: bool = True,
    ) -> ray.tune.ExperimentAnalysis:
        """
        Run Bayesian optimization

        Args:
            num_samples: Number of optimization trials
            n_cores: Number of CPU cores to use (default: number of CPU cores)
            experiment_name: Name for the experiment
            ray_config: Additional Ray configuration
            enable_checkpointing: Whether to enable checkpointing during optimization

        Returns:
            Ray Tune ExperimentAnalysis object with results
        """
        # Store num_samples in config for resource usage tracking
        self._config["num_samples"] = num_samples

        # Initialize Ray if not already done
        if not self._ray_initialized:
            ray_init_config = {
                "num_cpus": self._resource_config["num_cpus"],
                "num_gpus": self._resource_config["num_gpus"],
            }
            if ray_config:
                ray_init_config.update(ray_config)
            ray.init(**ray_init_config)
            self._ray_initialized = True

        # Setup algorithm
        algo = OptunaSearch(metric="score", mode="max")

        # Set concurrency limit
        if n_cores is not None:
            self.setup_distributed(n_cores=n_cores)
        else:
            # Auto-detect and use all cores minus 2
            self.setup_distributed()
        max_concurrent = self._resource_config["max_concurrent_trials"]
        algo = ConcurrencyLimiter(algo, max_concurrent=max_concurrent)

        # Setup scheduler
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            metric="score",
            mode="max",
            max_t=100,  # Maximum iterations per trial
            grace_period=1,  # Minimum iterations before early stopping
            reduction_factor=3,  # Factor by which to reduce tber of trials
        )

        if not self._configured:
            raise ValueError("Optimizer not configured. Use context manager first.")

        # Run optimization
        self.results = tune.run(
            self._objective_function,
            name=experiment_name,
            search_alg=algo,
            scheduler=scheduler,
            config=self._config["search_space"],
            num_samples=num_samples,
            resources_per_trial={"cpu": 1},
            storage_path=str(self.ray_results_dir),
        )

        return self.results

    def get_best_result(self) -> Dict:
        """
        Get the best optimization result

        Returns:
            Dictionary with best parameters and score
        """
        if self.results is None:
            raise ValueError("No optimization results available. Run optimize() first.")

        best_trial = self.results.get_best_trial("score", "max")
        return {
            "config": best_trial.config,
            "score": best_trial.last_result["score"],
            "trial_id": best_trial.last_result.get("trial_id"),
        }

    def get_results_dataframe(self):
        """
        Get optimization results as pandas DataFrame

        Returns:
            DataFrame with all trial results
        """
        if self.results is None and self._loaded_results is None:
            raise ValueError(
                "No optimization results available. Run optimize() or load_results() first."
            )

        if self._loaded_results is not None:
            return self._loaded_results

        return self.results.dataframe()

    def load_results(
        self, experiment_path: str, experiment_name: Optional[str] = None
    ) -> None:
        """
        Load results from a Ray Tune experiment directory

        Args:
            experiment_path: Path to Ray results directory
            experiment_name: Specific experiment name to load (optional)
        """
        experiment_path = Path(experiment_path)

        if not experiment_path.exists():
            raise FileNotFoundError(f"Experiment path not found: {experiment_path}")

        if experiment_name:
            # Load specific experiment
            exp_dir = experiment_path / experiment_name
            if not exp_dir.exists():
                raise FileNotFoundError(f"Experiment not found: {experiment_name}")
        else:
            # Find most recent experiment
            exp_dirs = [
                d
                for d in experiment_path.iterdir()
                if d.is_dir() and d.name.startswith("shipping")
            ]
            if not exp_dirs:
                raise FileNotFoundError("No experiments found in directory")
            exp_dir = max(exp_dirs, key=lambda x: x.stat().st_mtime)
            experiment_name = exp_dir.name

        # Load experiment analysis
        self.results = ray.tune.ExperimentAnalysis(str(exp_dir))
        self._loaded_experiment_name = experiment_name

        # Cache dataframe
        self._loaded_results = self.results.dataframe()

        print(f"Loaded experiment: {experiment_name}")
        print(f"Total trials: {len(self._loaded_results)}")

    def get_top_n_results(self, n: int = 5, metric: str = "score") -> List[Dict]:
        """
        Get the top N best results from the optimization

        Args:
            n: Number of top results to return
            metric: Metric to sort by

        Returns:
            List of dictionaries with top results
        """
        df = self.get_results_dataframe()

        # Sort by metric (assuming minimization)
        sorted_df = df.sort_values(by=metric).head(n)

        results = []
        for i, (idx, row) in enumerate(sorted_df.iterrows()):
            # Extract config parameters
            config = {}
            for col in df.columns:
                if col.startswith("config/"):
                    param_name = col.replace("config/", "")
                    config[param_name] = row[col]

            # Find associated planning file
            trial_id = row.get("trial_id", idx)
            planning_file = self.bayesopt_dir / f"{trial_id}.yml"

            results.append(
                {
                    "rank": i + 1,
                    "score": row[metric],
                    "trial_id": trial_id,
                    "config": config,
                    "planning_file": (
                        str(planning_file) if planning_file.exists() else None
                    ),
                }
            )

        return results

    def filter_results(
        self,
        score_threshold: Optional[float] = None,
        parameter_constraints: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Filter results based on score threshold and parameter constraints

        Args:
            score_threshold: Only include results better than this score
            parameter_constraints: Dict of parameter: (min, max) constraints

        Returns:
            Filtered DataFrame
        """
        df = self.get_results_dataframe().copy()

        # Filter by score
        if score_threshold is not None:
            df = df[df["score"] <= score_threshold]

        # Filter by parameter constraints
        if parameter_constraints:
            for param, (min_val, max_val) in parameter_constraints.items():
                col_name = f"config/{param}"
                if col_name in df.columns:
                    df = df[(df[col_name] >= min_val) & (df[col_name] <= max_val)]

        return df

    def export_results(
        self, format: str = "json", output_path: str = "./results"
    ) -> str:
        """
        Export optimization results in various formats

        Args:
            format: Export format (json, csv, yaml, excel)
            output_path: Output file path

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        df = self.get_results_dataframe()

        # Prepare data for export
        export_data = {
            "experiment_name": self._loaded_experiment_name or "current",
            "total_trials": len(df),
            "best_score": df["score"].max(),
            "timestamp": datetime.now().isoformat(),
        }

        if format == "json":
            if not str(output_path).endswith(".json"):
                output_path = output_path.with_suffix(".json")

            export_data["trials"] = df.to_dict(orient="records")
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)

        elif format == "csv":
            if not str(output_path).endswith(".csv"):
                output_path = output_path.with_suffix(".csv")
            df.to_csv(output_path, index=False)

        elif format == "yaml":
            if not str(output_path).endswith(".yml") and not str(output_path).endswith(
                ".yaml"
            ):
                output_path = output_path.with_suffix(".yml")

            export_data["trials"] = df.to_dict(orient="records")
            with open(output_path, "w") as f:
                yaml.dump(export_data, f)

        elif format == "excel":
            if not str(output_path).endswith(".xlsx"):
                output_path = output_path.with_suffix(".xlsx")

            with pd.ExcelWriter(output_path) as writer:
                # Summary sheet
                summary_df = pd.DataFrame([export_data])
                summary_df.to_excel(writer, sheet_name="Summary", index=False)

                # Results sheet
                df.to_excel(writer, sheet_name="Results", index=False)

                # Top 10 results
                top_10 = self.get_top_n_results(n=10)
                top_df = pd.DataFrame(top_10)
                top_df.to_excel(writer, sheet_name="Top 10", index=False)

        else:
            raise ValueError(f"Unsupported format: {format}")

        return str(output_path)

    def cleanup(self):
        """
        Clean up Ray resources
        """
        if self._ray_initialized:
            ray.shutdown()
            self._ray_initialized = False

    def __enter__(self):
        """
        Enter context manager and return config dictionary for modification

        Returns:
            Configuration dictionary that can be modified
        """
        return self._config

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context manager and finalize configuration
        """
        # Generate search space based on configuration
        if (
            self._config["use_planning_bounds"]
            and self._config["planning_results_file"]
        ):
            try:
                self._config["search_space"] = self._generate_bounds_from_file(
                    self._config["planning_results_file"], self._config["scale"]
                )
                print(
                    f"Generated search space from planning file with scale {self._config['scale']}"
                )
            except Exception as e:
                print(f"Warning: Failed to generate bounds from planning file ({e})")
                print("Using default search space")
                self._config["search_space"] = self._get_default_search_space()
        else:
            self._config["search_space"] = self._get_default_search_space()

        self._configured = True
